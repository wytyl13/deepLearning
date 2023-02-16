"""*******************************************************************************************
# Copyright (C) 2023. IEucd Inc. All rights reserved.
# @Author: weiyutao
# @Date: 2023-02-05 09:52:40
# @Last Modified by: weiyutao
# @Last Modified time: 2023-02-05 09:52:40
# @Description: we will learn deep learning in this program, this file is the first
# program for deep learning. we will use tensorflow to implement the concept about deep
# learning content in deep learning teaching material edit by Ian Goodfellow and
# youshua Bengio. the former author is the creater about Gan, the last author is his teacher.
# in order to insight into this program about deep learning, you should have a basic about
# deep learning, you should understand gradient descent, forward propagation, backward
# propagation and some optimization method for the deep neural network. advice you should
# learn about some introduction content made by Andrew Ng who is Ian Goodfellow's another teacher. 
# then, we will start our learning. 
# in this program, we will implement all deep learning concept via python, an interpreted language
# write used c language. why python? because we can quickly create the complex network used the tools
# what around the ecology of python. and the efficiency is also not low, just like the scientific computing
# in particular, the matrix operation, it is widely used in neural network. especially the dynamic matrix
# calculation, the efficient even more than the eigen in cpp if you used numpy, not to mension the 
# Mat class for opencv in cpp. and if you created the complex neural network, you will have a better choise, 
# tensorflow or pytorch, what is more efficient in building a network field than numpy. why more high efficiency?
# although their level is high, as is known to all, the higher-level language the lower hardware invocation cycle.
# it means the lower efficient. but this concept is just refer to the application program that local program.
# how to explain it, you can compare c and python, the former need to compiler first, then execute the bianary file.
# but python need not compiler first, you can read one code in python file and compiler it and execute.
# so the efficient for python program can not compare and c. but this efficient is just suitable the whole program.
# because of the generality between python and c, it makes we can code some high performance program and
# open interface for python, just like numpy, it is edited used c language, and used the strong application rules
# for hardware, just like the multiply threads, it can greatly improve the efficiency when you handle the dynamic
# matrix operation. of course, python call the interface about numpy is efficient, because it will
# run out of the limit about python language. so the lower efficient about python program does not mean
# that some library around python are also lower efficient. you can understand it as follow, python just a 
# script, you will call some other program in this script. there are a lot of the similar library, they are
# all open the interface for python, just like tensorflow, pytorch and so on.
# then, we will use tensorflow what a library to create the deep learning network, you can also use
# pytorch, it is better suitable for numpy, if you are a python programer or a new person to computer science, 
# you can use pytorch, it made you focus on deep learning not the computer code problem. we will select tensorflow in this program, 
# because I think it can let me better understand deep learning.  

# of course, we can also use the mature framework directly, but the skilled using the
# mature framework is not what we interested in. we intereted in those underlying theory
# about deep learning, so we will try to build the complex models by ourselves via the teaching
# material deep learning, then, we will try to understand how to construct complex neural network
# via tensorflow or pytorch. at last, we will consider how to call tensorflow in cpp program.
# at last, we will merge depth neural network and opencv. of course, we will construct our program
# used object-oriented feature in python.

# deep learning is original from 1940 to 1960, it is named as cybernetics.
# the second developlemnt is from 1980 s to 1990 s, it is named as connectionism.
# the third delelopment is from 2006 s, it is named deep learning.

# then, we will start our journey.
******************************************************************************************"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

from sample.imshow import Scatter
from sample.imshow import Line


class Algorithm:
    count = 0
    def __init__(self, name) -> None:
        self.name = name
        Algorithm.count += 1
    
    def printCount(self):
        print("total algorithm {}".format(self.count))
    
    def printAlgorithm(self):
        print("name, {}, function, {}".format(self.name, self.function))


"""
# @Author: weiyutao
# @Date: 2023-02-08 17:24:34
# @Parameters: 
# @Return: 
# @Description: 
"""


"""
# @Author: weiyutao
# @Date: 2023-02-09 11:03:10
# @Parameters: 
# @Return: 
# @Description: this is a class about linear regression.
# min Σi=0_m(e_i^2)
# ei = X_i@theta.T - y_i
# how to calculate the unknown param theta? derivative the expression.
# error = min Σi=0_m( X_i@theta.T - y_i)^2, notice, theta can be one number or one array list.
# the standard calculation method 1.
    calculate the partial derivatives for each unknown param. 
    just like error'_theta[0], error'_theta[1], you can get n theta unknown and n equation.
    to slove the equations.
# the standard calculation method 2. the normal equation method of linear regression.
# this method 2 means we want to calculate the derivative for the vector theta, not the theta of each samples.
# the former method is calculate the derivative of each sample J to theta. it is very complex.
# we will define the derivative based on the vector theta. we will get an expression what can calculate the theta directly,
# we need not to calculate the complex system of equations. 
    you should known the derivation about the matrix, vector and scalar. just like y = ax, 
    a is a vector, matrix, or a scalar, and x is a scalar. you will calculate the y'(x) = dy/dx = d(ax)/dx
    the dimension result will be equal to the dimension of x if the x is a scalar. we will descripe this result
    use the method as follow in order to conveninet to describe our problem.
    1 any dimension y, x is a scalar, the dimension of dy/dx is equal to the dimension of y.
    2 any dimension x, y is a scalar, the dimension of dy/dx is equal to the dimension of x.
    3 x, y are all the vector, one is row vector, another is column vetor, just like y(m, 1), 
        x(1, n), the dimension of dy/dx is equal to (m, n), dx/dy is also the same dimension.
    4 x, y are the same vector, all are row vector or column vector. just like x(1, m), y(1, n)
        or x(m, 1), y(n, 1), the dimension of dy/dx is euqal to (1, m) or (n, 1), the rule is 
        the row vector is equal to the dimension of x, the column vector is equal to the dimension of y.
    
    then, we can get the result of the least square method.
    dy(scalar)/dx(1, n), y is a scalar, x is a row vector. y = x@theta.T, theta is a row vector, x
    is also a row vector, so y is a scalar. so the dimension of dy/dx is equal to the dimension of x,
    it is a row vextor (1, n)

    we can get the standard least square method.
    J(theta) = 1/2m * ||X @ theta - y||^2  the expression ||X @ theta - y||^2 means the 2 norm about X@theta and y
    the dimension of X@theta is a vector, you can define it as a row vector or a column vector, it is not important.
    the important is its dimension must be euqal to the dimension of y. the value of 2 norm means the 
    distance between the two vectors. why is 1/2m not 1/m, the average should be 1/m, but we define the 1/2m
    in order to calculate the derivation result. then, our suppose will be equal to calculate the expression.J(theta)
    aim to get the minimize of the function J(theta)
    before making an calculation, we should understand the dimension of each variable.
    X(m ,n), y(m, 1), theta(n, 1)
    J(theta) = 1/2m * ||X @ theta - y||^2 = 1/2m*(X@theta - y).T @ (X@theta - y) ---(m, n)@(n, 1)-(m, 1) = (m, 1)---
    (m, 1).T@(m, 1) = (1, 1) is a scalar
    1/2m * (X@theta - y).T @ (X@theta - y) = 1/2m*(theta.T@X.T - y.T) @ (X@theta - y) = 
    1/2m*(theta.T@X.T@X@theta - theta.T@X.T@y - y.T@X@theta + y.T@y). then we can calculate the derivative d(J)/d(theta)
    dJ/d(theta) = 1/2m*(d(theta.T@X.T@X@theta)/d(theta) - d(theta.T@X.T@y)/d(theta) - d(y.T@X@theta)/d(theta) + d(y.T@y)/d(theta))
    you can get d(theta.T@X.T@X@theta)/d(theta) = (X.T@X+(X.T@X).T)@theta = 2*X.T@X@theta based on 
    (theta.T@A@theta)' = (A + A.T) @ theta. d(theta.T@X.T@y)/d(theta) = d[(X@theta).T@y]/d(theta) = 
    d[y.T @ (X @ theta)]/d(theta), d(y.T@y)/d(theta) = 0. so you can get the result about dJ/d(theta) = 
    1/2m*[2*X.T@X@theta - d(2*y.T@X@theta)/d(theta)] = 1/2m*(2*X.T@X@theta - 2*X.T@y)
    why d(2*y.T@X@theta)/d(theta) is not equal to 2*y.T@X, but is equal to 2*X.T@y?
    you can get the result based on the dimension. the dimension of (2*y.T@X@theta) is (1, n)@(m, n)@(n, 1) = (1, 1)scalar.
    the dimension of theta is a column vector (n ,1), the scalar derivative of vector is the vector itself.
    so you should ensure the result of derivative is (n, 1), but the dimension of 2*y.T@x is equal to (1, n)@(m, n)
    it is illegal matrix multiplication. but the dimension of 2*X.T@y is equal to (n, m) @ (m, 1) = (n, 1), the dimension is
    equal to the theta(n, 1), so it is the correct result of derivative. 

    the last result is d(J(theta))/d(theta) = 1/2m * ||X @ theta - y||^2 = 1/2m*(2*X.T@X@theta - 2*X.T@y)
    in order to get the minimize of the J function, you can make the d(J)/d(theta) is euqal to 0 to solve the
    value of unknown param theta.
    1/2m*(2*X.T@X@theta - 2*X.T@y) = 0 -> 2*X.T@X@theta = 2*X.T@y -> theta = [inverse of (X.T@X)] @ X.T @ y
    = (X.T@X)^-1@X.T@y.
    so you can just known X and y if you want to calculate the unknown param theta.

# the third mthod is grandient descent, it means you should init one random scalar or one array list for theta.
# you can init 0 for the theta. then, you should calculate the cost function based on each theta. you should update
# the theta based on the grandient descent. what is grandient descent? the cost function looks like the backward of a mountain, 
# you want to walk from top to bottom and waste less time, you should select a correct and fast direction. the slope is the correct
# direction, but the slope point the fast direction to up the hill, so you should use the negative slope.
# then, you should adjust the direction that down the hill for each step. and you should define a step length for
# it is alpha in this case, it means learning rate. you should select the suitable learning rate, too big will
# influence the results of study, too small will influence the learning efficiency. you should use the new theta to calculate
# the cost function value for each step. and you can set the learning iters to end your learning process. or you will
# must to select a standard condition to end your process after the suitable times.
# you can calculate each dirivative of each J(theta), J(theta_i) = 1/2m*(X^[i]@theta_i-yi)^2, i is range from 0 to m.
# J(theta_i)' = 1/m*(X^[i]@theta_i-y_i)@X.T, this slope, you should update theta used it.
# theta_i -= alpha * 1/m*(X^[i]@theta_i-y_i)@X.T
# the least square method. 1/2m * Σi=0_m(X_i @ theta - y_i)^2
# caluclate the mean of all the cost of samples
"""
class LinearRegression(Algorithm):

    # you should pass the param used inputData=X, inputLabel=y in the construct function.
    # you will fail to pass the param if you do not use this method.
    def __init__(self, name, inputData, inputLabel) -> None:
        super().__init__(name)
        self.inputData = inputData
        self.inputLabel = inputLabel
    

    """
    # @Author: weiyutao
    # @Date: 2023-02-09 11:42:09
    # @Parameters: inputdata(m, n), inputlabel(m, 1), theta(n, 1)
    # @Return: 
    # @Description: you can use matrix or ndarray type to matrix calculations.
    # but the difference between matrix and ndarray is the former has the transpose method, 
    # the last type ndarray has not this method. the ndarray is the generally data type.
    # notice, the default parmeters can not use the self. as the default value.
    # because it has not the type statement in python, so you should define the class in the function,
    # you should not use the method or attribution if you do not define the class in this function.
    """
    def cost(self, theta):
        error = self.inputData @ theta - self.inputLabel
        inner = np.power(error, 2)
        return np.sum(inner) / (2 * len(self.inputData))
    
    """
    # @Author: weiyutao
    # @Date: 2023-02-09 16:28:21
    # @Parameters: 
    # @Return: 
    # @Description: theta_i -= alpha * 1/m*(X^[i]@theta_i-y_i)@X.T, the important for gradient descent
    # is how to select the most correct param, involved theta, alpha, iter and so on. we have some problem
    # for the gradinet descent.
    # 1 the smaller cost will get the most correct result? no.
    # 2 the smaller alpha will get the most correct result? yes, the smaller alpha wii result to the smaller 
    # shock. the smaller shock will result to the higher accuracy. but it also result to the lower efficiency.
    # 3 the bigger iters will result to the higher accuracy? yes, it is equal to the alpha, it will affect your 
    # learning efficiency directly.
    # 4 how to select the correct params? you should adjust the param based on the result.
    """
    def gradientDescent(self, theta, alpha, iters, isPrint = False):
        X = self.inputData
        y = self.inputLabel
        costs = []
        for i in range(iters):
            # you can not use theta -=, it is not meaningful in python.
            # you will creat a new variable theta as follow. you'd better use this method
            # to update the param theta. or you will get the error what you can not understand the reason for the error.
            theta = theta - ((X.T @ (X @ theta - y)) * alpha / len(X))
            cost = self.cost(theta)
            costs.append(cost)
            if(i % 100 == 0):
                if(isPrint):
                    print(cost)
        return theta, costs

    """
    # @Author: weiyutao
    # @Date: 2023-02-09 15:47:29
    # @Parameters: X(m, n), y(m, 1)
    # @Return: theta(n, 1)
    # @Description: theta = [inverse of (X.T@X)] @ X.T @ y = (X.T@X)^-1@X.T@y.
    """
    def normalEquation(self):
        X = self.inputData
        temp = X.T
        theta = np.linalg.inv((temp @ X)) @ temp @ self.inputLabel
        return theta

    """
    # @Author: weiyutao
    # @Date: 2023-02-10 17:50:21
    # @Parameters: 
    # @Return: 
    # @Description: you should pass the theta what you have optimized. 
    """
    def predict(self, test, theta):
        predict = test @ theta
        return predict
    
    def accuracy(self, test_X, test_y, theta):
        predict = self.predict(test_X, theta)
        accuracy = np.mean(np.fabs(test_y - predict))
        return accuracy
    
    """
    # @Author: weiyutao
    # @Date: 2023-02-10 22:53:12
    # @Parameters: 
    # @Return: 
    # @Description: show predict function involved predict and fitting the regression line.
    """
    
    def imshow_predict(self, theta):
        if(type(self.inputData).__name__ == 'ndarray'):
            X = self.inputData
        else:
            X = self.inputData.getA()
        prob = self.predict(theta)
        fig, ax = plt.subplots()
        plt.scatter(X[:, :1], self.inputLabel, color='blue', marker='o', label='original data')
        plt.plot(X[:, :1], prob, color='red', linestyle='--', label='predict value')
        ax.set_xlabel(xlabel='the feature 1', fontsize=18)
        ax.set_ylabel(ylabel='the label', fontsize=18)
        ax.set_title(label=self.name, fontsize=18)
        plt.legend(loc = 2)
        plt.show()

# you should nitice the difference between -> none and not any content.
class LogisticRegression(LinearRegression):
    def __init__(self, name, inputData, inputLabel) -> None:
        super().__init__(name, inputData, inputLabel)

    # sigmoid function:
    # 1/1+e^(-z)
    # this is a simple transform from continuous variables to discrete probability variable.
    # notice, log(0) is meaningless, so you should ensure z value, it can not too big or small.
    # if z is very small, sigmoid function will return 0. log(0) is meaningless, error will happen.
    # the return value x@theta can not too small.
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    """
    # @Author: weiyutao
    # @Date: 2023-02-09 18:22:01
    # @Parameters: theta(n, 1)
    # @Return: 
    # @Description: 
    # rewriting, you can rewrite function directly in python.
    # but python it is not supported for overloading in python.
    # you need not to do anything if you do not want to overloading or rewriting.
    # you can call the function of father used your instance.
    # J(theta) = -1/m[y * ln(sigmoid(z)) + (1 - y) * ln(1 - sigmoid(z))]
    # d[1/(1+e^-X@theta)]/d(theta) = -(1+e^-X@theta)^-2*[d(X@theta)/d(theta)] -->[[d(X@theta)/d(theta)] = d(n, 1)/d(n, 1) = (n, 1)]
    # J(theta)' = -1/m*(y*)
    # X.T @ (A - y)
    # we should add regularization if we consider it. it will reduce
    # the fitting phenomenon. of course, you should consider the regularization
    # in cost function and gradientDescent function. the regularization is added into
    # the cost function, so it will influence the regression result. because the 
    # aims what we want to get is minimize the cost. we have used gradientDescent method
    # to find the minimize cost.
    # reg = lamda / 2m * theta^2
    # so J(theta) = 1/2m*Σi=1_m(sigmoid(X_i@theta_i - y_i)^2) + lamda/2m * Σj=1_m(theta_j)^2, this is line regression
    # J(theta) = -1/m[y * ln(sigmoid(z)) + (1 - y) * ln(1 - sigmoid(z))] + lamda/2m * Σj=1_m(theta_j)^2
    # you should distinguish the line regression and logistic regression.
    # the former used least square multiplication. the last is not use it.
    """
    def cost(self, theta, lamda=0):
        X = self.inputData
        y = self.inputLabel
        # theta(n+1, 1), (1, n)@(n, 1) = (1, 1)
        reg = theta[1:, :].T @ theta[1:, :] * (lamda / (2 * len(X)))
        # A(m, 1), first(1, 1), second(1, 1)
        A = self.sigmoid(X @ theta)
        first = y.T @ np.log(A)
        second = (1 - y).T @ np.log(1 - A)
        return -np.sum(first + second) / len(X) + reg
    
    """
    # @Author: weiyutao
    # @Date: 2023-02-13 12:51:30
    # @Parameters: X(m, n+1), y(n+1, 1), theta(n+1, 1), alpha(float scalar), m(int scalar, m)
    # iters(int scalar), lamda(float scalar).
    # @Return: 
    # @Description: 
    """
    def gradientFunction(self, X, y, m, theta, alpha, iters, isPrint=False, lamda=0):
        costs = []
        for i in range(iters):
            # because the intercept coe don't as regularization
            # dJ/d(theta) = 1/m(X.T@(A-y))
            # consider regularization, 1/m(X.T@(A-y)) + 1/m*theta, notice, theta just involved 1:, because
            # do not consider the intercept. consider the lamda and alpha.
            # alpha * lamda * [1/m(X.T@(A-y)) + 1/m*theta]
            # in order to use the broadcast in python, we must insert one row for reg. because the operation is add.
            # so we will add zero as the first row for reg.
            reg = alpha * theta[1:, :] * lamda / m
            reg = np.insert(reg, 0, values=0, axis=0)
            # reg(n+1, 1), A(m, 1), X(m, n+1), X.T @ (A - y) = (n+1, 1) = first
            # theta(n+1, 1)
            A = self.sigmoid(X @ theta)
            first = (X.T @ (A - y)) * alpha / m
            theta = theta - (first + reg)
            if (i % 10000 == 0):
                cost = self.cost(theta, lamda)
                costs.append(cost)
                if isPrint:
                    print(cost)
        return theta, costs

    """
    # @Author: weiyutao
    # @Date: 2023-02-10 15:55:52
    # @Parameters: theta(n, 1), alpha(a scalar), iters(int number)
    # param k is the class numbers.
    # @Return: theta_all, you can get the theta_all dimension is (n+1, k) if you pass the k != 0.
    # you will get the theta_all dimension is (n+1, 1) if you pass the k = 0. the former theta means
    # k numbers theta, each theta means the current index as 1 label, the other index is 0 label.
    # so you can use the theta_all to predict multiple classific problem. you can use X @ theta_all, 
    # predict = (m, n+1) @ (n+1, k) = (m, k), just like predict[:, :1], the label will be 1 if the probability is 
    # greater than 0.5, or the label will be 0, and the mapping classific is the index value. so the label 1 will
    # be classified 0, the second column label 1 will be 1, the third column label 1 will be 2.
    # @Description: you should distinguish the bianry logistic and multiple logistic.
    # the different between them as follow:

    # first, how to classification? binary logistic can separate different class by comparing the probability
    # batween one class and another class, the label will be one if the probability is greater than 0.5. or
    # the label will be zero. multiple logistic can use the same method. you can compare the probability between one class 
    # and other class, in this cycle, you will find all different class by compare the probability of current class
    # and other class. so this is a main difference between binary logistic and multiple logistic.

    # second, you should distinguish the difference between linear separable and linear inseparable.
    # in order to handle the linear inseparable problem, you should consider two method ad least.
    # one, you should consider the polynomial regression.
    # two, you should consider the regularization what can reduce overfitting.
    # three, you should cosider the method to reduce the underfitting. of course, the regularization can
    # also handle the underfitting problem.
    # in order to deep understand the overfitting and underfitting, we will introduction two indicators, 
    # just like the mean and variance gray value in digital image process, we want to introduct the
    # bias and variance, the bias means the gap between predict and real value, it show the fitting ability
    # about the algorithm itself. the variance means the range of predictive value. so we can conclude that
    # the higher bias will result to underfitting, the higher variance will result to the overfitting problem.
    # so we can improve the accuracy of the model by  considering to adjust these two indicators.
    # we have got the regularization
    # in the former function cost, we will consider it in the gradientDescent function.
    # J(theta) = -1/m[y * ln(sigmoid(z)) + (1 - y) * ln(1 - sigmoid(z))] + lamda/2m * Σj=1_m(theta_j)^2
    # dJ/d(theta) = (1/m*Σ(X_i@theta-y)@X.T) + lamda/m*theta, you can find the derivative expression of
    # dJ/d(theta) for the logistic is same to the linear regression. the different is logistic regression
    # run the sigmoid function. although the cost function is different between them. you can also find
    # the role of regularization during the grandientDescent, theta -= [(1/m*Σ(X_i@theta-y)@X.T) + lamda/m*theta]
    # the bigger lamda, the faster falling speed for theta unknown number. and the regularization coefficient contains
    # the theta unknown number itself. you can transform the expression by extract lamda param.
    # theta = theta*(1-alpha*lamda/m) - alpha*1/m(X@theta-y)@X.T, the expression about update theta param
    # that dose not contain the regularization is theta = theta - alpha[(1/m*Σ(X_i@theta-y)@X.T)
    # you can find the difference is to scale the theta. the bigger lamda, the smaller theta. what it menas?
    # if you set a large enough lamda, the theta will tend to be zero. the second part of the expression can be
    # ignored, because it just means the direction, and the alpha is small enough to can be ignored.
    # so the bigger lamda, the fitting curve you got will be a line. it means you will just consider the intercept
    # coefficient in your theta param, the other coefficient in theta has ignored, why?because we does not
    # consider the regularization into the intercept. so we can conclude that the lamda can not be too big, 
    # of course, it can not be too small, or you will get the original result that does not contain the regularization.
    # 
    # the regularization coefficient will influence the model result based on the expression in former line.
    # you can find the different between the dJ/d(theta) contains the regularization and dose not contains the regularization.
    # 
    # then, we will modify the gradientDecent function based on the concept above.
    # of course, we will set the function as far as possible generally.
    # first, we should add lamda. second, consider the multiple logistic regression.
    # last, consider to add the polynomial.
    # we can use overload in python, but you must to do something, because the overload does not be supported
    # in python like cpp and java. you can implement the function add *args param into python function.
    # it will be without limiting the param numbers, and you can get all param you have passed use it.
    # you can use the attribution len(args) to set the different function. but it is not suitable for the conditon
    # when you want to pass the different type data. you can with the help of other module in python.

    """
    def gradientDescent(self, theta, alpha, iters, isPrint=False, lamda=0, k=0):
        X = self.inputData
        y = self.inputLabel
        m = len(X)
        # n = 3
        n = X.shape[1]
        # if k != 0, it means we will meet a multiple logistic, the method involved
        # one vs rest, or one vs one.
        if k != 0:
            # ensure the dimension is euqal to the dimension of theta that we have defined in function.
            # theta_all(3, 3)
            theta_all = np.zeros((k, n)).T
            # you will get k list, [1, 0...], [0, 1, ...], [0, 0, 1, ...]
            # 
            for i in range(k):
                y_classify = []
                theta_ = np.ones((n, 1))
                theta = theta_.reshape(theta_.shape[0], 1)
                # theta(3, 1)
                for j in y:
                    if j == i:
                        y_classify.append(1)
                    else:
                        y_classify.append(0)
                # ensure the dimension is equal to the function gradinetFunction that we have defined.
                y_classify = np.array(y_classify).reshape(len(y_classify), 1)
                # y_classify(150, 1)
                # then, you can start to grandientDescent used the classsify y_classify
                theta, costs = self.gradientFunction(X, y_classify, m, theta, alpha, iters, isPrint, lamda)
                # theta(3, 1), before set (3,1) to (3,), you should transform the dimension first.
                # ravel function can transform (3, 1) to (3, )
                theta_all[:, i] = theta.ravel()
        if k == 0:
            theta_all, costs = self.gradientFunction(X, y, m, theta, alpha, iters, isPrint, lamda)
        return theta_all, costs

    """
    # @Author: weiyutao
    # @Date: 2023-02-10 17:36:59
    # @Parameters: test(test_m, n+1), theta(n+1, 1) or (n+1, k), prob(test_m, 1) or (test_m, k)
    # @Return: 
    # @Description: prob is predict value, this value is probability, so we should transform
    # the probabily to true value, if the probabily is greater than 0.5, the true value is 1, 
    # or the true value is 0. if you want to use the same function predict inheritanced from father,
    # you need not to define the same function. you should use super.predict(). notice, ndarray data type
    # is greater suitable for the matplotlib than matrix. so we generally transform the inputdata from any 
    # type to ndarray type used getA() function when we want to matplotlib this data.
    # notice. you should consider sigmoid function when you predict the logistic regression problem.
    """
    def predict(self, test, theta):
        prob = self.sigmoid(test @ theta)
        prob = np.around(prob, 3)
        m = len(test)
        k = prob.shape[1]
        predict_value = np.zeros((m, k))
        if(k != 1):
            for i in range(k):
                predict_value[:, i] = [i if x >= 0.5 else 0 for x in prob[:, i]]
            predict_value = np.max(predict_value, axis=1)
        if(k == 1):
            # for i in range(m):
            #     if prob[i:i+1, :] >= 0.5:
            #         predict_value[i] = 1
            #     else:
            #         predict_value[i] = 0
            predict_value = [1 if x >= 0.5 else 0 for x in prob]  
        # predict = np.array(predict_value).reshape(m ,1)
        return prob, predict_value

    def accuracy(self, test_m, test_y, theta):
            _, predict_value = self.predict(test_m, theta)
            predict_value = predict_value.reshape(len(test_m), 1)
            return np.mean(test_y - predict_value)

    """
    # @Author: weiyutao
    # @Date: 2023-02-10 19:09:30
    # @Parameters: 
    # @Return: 
    # @Description: you should distinguish linear separable and linear inseparable.
    # linear separable can plot one line to separate the samples.
    # the linear inseparable should use area or circle to separate the sample
    # the decision boundary of linear separable expression is y = (-theta_1 - theta_2*x) / theta_0
    # theta_0 is the intercept coefficient, theta_1 to theta_2 is the coefficient of each features. 
    # x is the x axis numbers.
    # this show predict function about logical regression involved predict and draw the decision boundary.
    # the decision boundary involved linear and nonlinear decision boundary.
    # it is not equal to the fitting regression, the fitting value for linear regression 
    # is predict value, but the predic value for logical regression is a discrete label.
    # so you should define the expression about decision boundary for logical regression.
    # we will consider the linear decision boundary. we have optimized the coefficient theta.
    # theta_0 is the intercept item coefficient, the other is the coefficient of each feature.
    # we can define the linear decision boundary, X(m, n+1)@theta(n+1, 1) = 0(m, 1), if this expression
    # is equal to 0, of course we can get a (m, 1) dimension result. if the result is 0, then it will be
    # the linear decision boundary. if the result of one sample is greater than 0, the predict label of 
    # the sample is 1, or the predict label will be 0. then, we can get another type based on the expression.
    # theta_0 + x_1 * theta_1 + x_2 * theta_2 = 0, x2 = (-theta_0 - x_1 * theta_1) / theta_2
    # this expression suitable for the samples, of course it is also suitable for any value, because
    # it is a line function in coordinate system.
    """
    def imshow_predict(self, theta):
        x_label = np.arange(4, 8, step=0.1)
        y_label = (-theta[0] - x_label * theta[1]) / theta[2]
        scatter = Scatter("scatter figure")
        scatter.imshow(self.inputData[:, 1:3], x_label, y_label)