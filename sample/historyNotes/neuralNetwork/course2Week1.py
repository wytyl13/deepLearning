# -*- coding: utf-8 -*-
#*********************************************************************************************#
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/7/19 8:54:49
#   @File Name : course2Week1.py
#   @Description :首先，本文件的内容；初始化参数（分别使用0和随机数来初始化参数看效果，最后使用抑梯度异常来初始化参数）、正则化模型（使用二范数对二分类模型正则化防止过拟合、使用随机删除节点的方法精简模型尝试避免过拟合）和梯度校验（对模型使用梯度校验，检测它是否在梯度下降的过程中出现误差过大的情况） 
#**********************************************************************************************#

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
#以下是一种方法解决导入包和被导包不再一个路径下的问题。还有一种方法就是在被导包及其以上的目录下添加_init_.py文件。注意这个文件可以是空白文件，但是必须有，有了以后直接使用路径.的方式导入即可,也就是说如果你想直接使用该文件名称导入，那么在该文件的同级目录下必须有_init_.py文件。但是要注意的是，启示目录必须要和当前编辑的文件属于平级目录，否则不能实现.
import sys
sys.path.append("../data/course2Week1/")

import init_utils
import reg_utils
import gc_utils

#%matplotlib inline

plt.rcParams['figure.figsize'] = [7.0,4.0]
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.show()


#netual net model
def model(X,Y,learning_rate = 0.01,num_iterations = 15000,print_cost = True,initialization = "he",is_plot = True):
    #then,we would creat a L3 netual net
    #linear->relu -> linear->relu -> linear->sigmoid
    #parameters:
        #X:the train data or test data
        #Y:label,the 0 or 1.
        #initialization:the initial type
    
    grads = {}
    costs = []
    m = X.shape[1]
    #the layers numbers,the define of the layers is aimed to initialize the parameters.
    layers_dims = [X.shape[0],10,5,1]
    
    #conclude the input type of initial parameters
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    else:
        print("input is erro!exit")
        exit

    for i in range(0,num_iterations):
        #forward:here we need deep learning again
        a3,cache = init_utils.forward_propagation(X,parameters)
        cost = init_utils.compute_loss(a3,Y)
        #backward:here we need deep learing again
        grads = init_utils.backward_propagation(X,Y,cache)
        #optimize the parameter
        parameters = init_utils.update_parameters(parameters,grads,learning_rate)

        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print("the " + str(i) + "th,cost is" + str(cost))

    
    if is_plot:
        plt.plot(costs)
        plt.ylabel('costs')
        plt.xlabel('iteration (per hundred)')
        plt.title('Learning rate = ' + str(learning_rate))
        plt.show()
    #return the result parameters
    return parameters

#define the initialize_parameters function
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for i in range(1,L):
        parameters["W" + str(i)] = np.zeros((layers_dims[i],layers_dims[i - 1]))
        parameters["b" + str(i)] = np.zeros((layers_dims[i],1))
    return parameters        


#define the random initialization
def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
    return parameters

#test the predict decision boundary
def plot_boundary(initialization,train_X,train_Y,parameters,left = -1.5,right = 1.5):
	fig,ax = plt.subplots()
	ax.set_title(label = "Model with " + str(initialization) + " initialization")
	ax.set_xlim(left,right)
	ax.set_ylim(left,right)
	init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters,x.T),train_X,train_Y)

#inhibition of the gradient initialization
#ramdom the sqrt(2 / layers_dims[l - 1])
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l  -1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
    return parameters




if __name__ == "__main__":
    
    #load image dataset
    #notice:the type of the train_X and train_y all is numpy array.the index 0 is the n,the index 1 is m
    train_X,train_Y,test_X,test_Y = init_utils.load_dataset(is_plot = True)

    '''
    #layers_dims = [train_X.shape[0],10,5,1]
    parameters = model(train_X,train_Y,initialization = "zeros",is_plot = True)
    #print the predict use the parameters
    prediction_train = init_utils.predict(train_X,train_Y,parameters)
    prediction_test = init_utils.predict(test_X,test_Y,parameters)
    print(prediction_train)
    fig,ax = plt.subplots()
    ax.set_title(label = 'Model with Zeros initialization')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters,x.T),train_X,train_Y)
    '''


    ''' 
    #test the random initialize
    parameters = model(train_X,train_Y,initialization = 'random',is_plot = True)
    #here we can find that the zeros initialize is less useful than random initialize.
    plot_boundary('random',train_X,train_Y,parameters)

    #here we can find the cost start to decline,but the cost value is very big when start,so random need to update.the update thought is initialize the smaller weight as the initialization weight
    prediction_train = init_utils.predict(train_X,train_Y,parameters)
    prediction_test = init_utils.predict(test_X,test_Y,parameters)
    print(prediction_train,"\n",prediction_test)
    '''

    #test the random inhibition gradient initialization
    parameters = model(train_X,train_Y,initialization = 'he',is_plot = True)
    plot_boundary('he',train_X,train_Y,parameters)
    prediction_train = init_utils.predict(train_X,train_Y,parameters)
    prediction_test = init_utils.predict(test_X,test_Y,parameters)
    print(parameters) 


