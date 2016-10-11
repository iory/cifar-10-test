#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals

from time import time
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers


class DNN(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv1=F.Convolution2D(3, 32, 3, pad=1),
            conv2=F.Convolution2D(32, 32, 3, pad=1),
            conv3=F.Convolution2D(32, 32, 3, pad=1),
            conv4=F.Convolution2D(32, 32, 3, pad=1),
            conv5=F.Convolution2D(32, 32, 3, pad=1),
            conv6=F.Convolution2D(32, 32, 3, pad=1),
            l1=F.Linear(512, 512),
            l2=F.Linear(512, 10))
    def __str__(self):
        return "#<{0.__class__.__name__}>".format(self)
    def __repr__(self):
        return self.__str__()
    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 2)
        h = F.dropout(F.relu(self.l1(h)), train=train)
        y = self.l2(h)
        return y

def unpickle(f):
    import pickle
    fo = open(f, 'rb')
    d = pickle.load(fo)
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

    return train_data, test_data, train_target, test_target

train_X, test_X, train_y, test_y = load_cifar10("data")
train_X = train_X.reshape((len(train_X), 3, 32, 32))
test_X = test_X.reshape((len(test_X), 3, 32, 32))

gpu = 0
epochs = 1
batchsize = 100
out = 'result'
model = L.Classifier(DNN)
chainer.cuda.get_device(0).use()
model.to_gpu()

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

train_iter = chainer.iterators.SerialIterator(zip(train_X, train_y), batchsize)
test_iter = chainer.iterators.SerialIterator(zip(test_X, test_y), batchsize,
                                             repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
trainer = training.Trainer(updater, (epochs, 'epoch'), out=out)

# Take a snapshot at each epoch
trainer.extend(extensions.snapshot())

trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy']))

trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.LogReport())

# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

trainer.run()
