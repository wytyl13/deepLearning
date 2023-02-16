#!/usr/bin/env python
"""
# Author: weiyutao
# Created Time : 2022/7/29 8:55:18
# File Name: 
# Description:minibatch
    here we also test the update of the parameters
    first,we will use the SGD
    then,we will use the mini-batch
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
import sys
sys.path.append("../data/course2Week1/")
import opt_utils
import testCase


def update_parameters_with_gd(parameters,grads,learning_rate):
    #the parameters involved the W and b 
    L = len(parameters) // 2#the layers of neutual network
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(L + 1)] - learning_rate * grads["dW" + str(L + 1)]
        parameters["b" + str(L + 1)] = parameters["b" + str(L + 1)] - learning_rate * grads['db' + str(L + 1)]
    
    return parameters

if __name__ == "__main__":
    parameters,grads,learning_rate = testCase.update_parameters_with_gd_test_case()
    parameters = update_parameters_with_gd(parameters,grads,learning_rate)
    print(parameters)
