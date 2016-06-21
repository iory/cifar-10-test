#!/bin/sh -eu

FILE_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p $FILE_DIRECTORY/../data
cd $FILE_DIRECTORY/../data

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xvzf cifar-10-python.tar.gz
