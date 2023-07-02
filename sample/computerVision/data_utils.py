'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-05-07 19:15:59
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-05-07 19:15:59
 * @Description: this file is to load the datasets CIRFI10 for the classifier application.
***********************************************************************'''
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if(version[0] == '2'):
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))



def load_CIFAR_batch(fileName):
    """ load one batch data, there are five batches, one batch has 10000 samples """
    with open(fileName, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all samples """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" %(b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtrain = np.concatenate(xs)
    Ytrain = np.concatenate(ys)

    del X, Y
    Xtest, Ytest = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    print(type(Xtrain), type(Ytrain), type(Xtest), type(Ytest))
    return Xtrain, Ytrain, Xtest, Ytest

def imshowMatrix(dists):
    plt.imshow(dists, interpolation='none')
    plt.show()

