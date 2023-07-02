"""******************************************************************************************
# Copyright (C) 2023. IEucd Inc. All rights reserved.
# @Author: weiyutao
# @Date: 2023-02-06 13:10:32
# @Last Modified by: weiyutao
# @Last Modified time: 2023-02-06 13:10:32
# @Description: package is a directory for python. moudle is a py file in the package.
# just like import numpy in your program, numpy is a moudle, python program will find the module
# in your sys.moudles what involved all modules you ever used in your system. you can import sys to
# print this attribute. if program can not find it in sys.modules. python will find them in
# standard lib, then will go to sys.path. generally, your program directory will be stored in
# sys.path attribute. you can import a package or a module. import package means to import 
# the __init__.py file in the package. you can also import a specific class or function.
# you should separate write the import code. involved import standard lib, three party modules,
# and import yourself modules. you can import or from ... import ..., they all have the same effect.
******************************************************************************************"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sample.general.general import LinearRegression
from sample.general.general import LogisticRegression
from sample.general.imshow import Imshow
from sample.general.imshow import Scatter
from sample.general.imshow import Line
from sklearn.linear_model import LogisticRegression as Logistic


if __name__ == "__main__":
    # test linear regression
    """     
    X = np.random.randint(0, 5, (100, 2))
    y = X[:, :1] * 2 + X[:, 1:]
    # X = np.matrix(inputData)
    # y = np.matrix(inputLabel)
    theta = np.array([0, 0])
    theta = theta.reshape(theta.shape[0], 1)
    print(theta.shape)
    linearRegression = LinearRegression("line regression", X, y)
    cost = linearRegression.cost(theta)
    print(cost)
    theta, cost = linearRegression.gradientDescent(theta, 0.1, 1000, True)

    theta_norm = linearRegression.normalEquation()
    print(theta)
    print(theta_norm)  
    linearRegression.imshow_predict(theta) 
    """


    # test imshow line and picture. 
    """ 
    X = iris.data[:, :2]
    line = Line("line figure")
    line.imshow(X)

    path = ""
    image = plt.imread("hln.png")
    picture = Imshow('picture figure')
    picture.imshow(image) 
    """

    # test logical regression
    # you should define some coordinates and binary 0, 1 to test this case. we will generate some original data
    # here we will use the official datasets in sklearn module.
    iris = load_iris()
    # print(iris['data'].shape) 
    # the features of iris has 4, 150 samples. we just need to two features.
    # so we index former two features in the data sources.
    # theta = np.matrix(np.array([0, 0, 0])).T
    theta = np.ones((3, 1))
    X_ = iris.data
    index_0 = np.ones(len(X_))
    X = np.c_[index_0, X_]
    m = iris.target
    y = m.reshape(m.shape[0], 1)
    scatter = Scatter("scatter figure")
    scatter.imshow(X[:, 1:3])
    # test the binary logistic regression. used two features, and set the intercept, so we should
    # index the former three columns data from X. and index the former 100 rows data from y.
    # it can show the binary classific, zero and one is the label, there is 50 samples for each label.

    y1 = np.ones(50)
    y2 = np.zeros(50)
    y3 = np.zeros(50)
    y_1 = np.concatenate((np.concatenate((y1, y2), axis=0), y3), axis=0).reshape(150, 1)
    y_2 = np.concatenate((np.concatenate((y2, y1), axis=0), y3), axis=0).reshape(150, 1)
    y_3 = np.concatenate((np.concatenate((y2, y3), axis=0), y1), axis=0).reshape(150, 1)

    y_4 = np.concatenate((np.concatenate((y_1, y_2), axis=1), y_3), axis=1).reshape(150, 3)
    print(y_4)

    # logisticRegression = LogisticRegression("binary logistic regression", X[:100, :3], y[:100, :])
    # theta, costs = logisticRegression.gradientDescent(theta, 0.1, 200000, True, 0.1)
    # prob, predict = logisticRegression.predict(X[:100, :3], theta)
    # logisticRegression.imshow_predict(theta)  
    
    # test the multiple logistic regression. you can also use regularization coefficiency.
    logisticRegression1 = LogisticRegression("multiple logistic regression", X[:, :3], y_1)
    logisticRegression2 = LogisticRegression("multiple logistic regression", X[:, :3], y_2)
    logisticRegression3 = LogisticRegression("multiple logistic regression", X[:, :3], y_3)
    theta1, _ = logisticRegression1.gradientDescent(theta, 0.01, 8000, False, 0, 0)
    theta2, _ = logisticRegression2.gradientDescent(theta, 0.01, 8000, False, 0, 0)
    theta3, _ = logisticRegression3.gradientDescent(theta, 0.01, 8000, False, 0, 0)


    theta__ = np.concatenate((np.concatenate((theta1, theta2), axis=1), theta3), axis=1)
    logisticRegression = LogisticRegression("multiple logistic regression", X[:, :3], y)
    theta, _ = logisticRegression.gradientDescent(theta, 0.01, 200000, False, 0, 3)
    # prob, predict_value = logisticRegression.predict(X[:, :3], theta)
    print(logisticRegression.accuracy(X[:, :3], y, theta))







    



