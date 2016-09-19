#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals

import argparse
import os

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

###################
# logger settings #
###################
import logging
import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s]:[%(asctime)s]: %(message)s'))
logger = colorlog.getLogger('example')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            conv1=F.Convolution2D(3,32,3),
            bn1 = F.BatchNormalization(32),
            conv2 = F.Convolution2D(32, 64, 3, pad=1),
            bn2 = F.BatchNormalization(64),
            conv3=F.Convolution2D(64,64,3,pad=1),
            fl4=F.Linear(1024, 256),
            fl5=F.Linear(256, 10)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), 2)
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), 2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), 2)
        h = F.dropout(F.relu(self.fl4(h)))
        y = self.fl5(h)
        return y

def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def load_cifar10(datadir):
    train_data = []
    train_target = []

    # 訓練データをロード
    for i in range(1, 6):
        d = unpickle("%s/data_batch_%d" % (datadir, i))
        train_data.extend(d["data"])
        train_target.extend(d["labels"])

    # テストデータをロード
    d = unpickle("%s/test_batch" % (datadir))
    test_data = d["data"]
    test_target = d["labels"]

    # データはfloat32、ラベルはint32のndarrayに変換
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.int32)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.int32)

    # 画像のピクセル値を0-1に正規化
    train_data /= 255.0
    test_data /= 255.0

    train_data = train_data.reshape(len(train_data), 3, 32, 32)
    test_data = test_data.reshape(len(test_data), 3, 32, 32)
    train_target = np.array(train_target)
    test_target  =np.array(test_target)

    return zip(train_data, train_target), zip(test_data, test_target),

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit, 10))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = load_cifar10(os.path.join(os.path.dirname(__file__), '../data/cifar-10-batches-py'))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
