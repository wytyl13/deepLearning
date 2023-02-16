#!/usr/bin/env python
# -*- coding=utf8 -*-
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
import course2Week1 as cw

def update_parameters_with_gd(parameters,grads,learning_rate):
    #the parameters involved the W and b 
    L = len(parameters) // 2#the layers of neutual network
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    
    return parameters


#define miniBatch
def random_mini_batches(X,Y,mini_batch_size = 64,seed = 0):
    
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    
    
    #step 1 random sort
    #random sort the X and Y using the same random list permutation
    #generate the list,index numbers is m
    permutation = list(np.random.permutation(m))
    #sort the column of X based on the list permutation
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((1,m))

    #step 2 split based on the mini_batch_size
    miniBatchSizeNum = math.floor(m / mini_batch_size)#take up the whole
    for i in range(0,miniBatchSizeNum):
        mini_batch_X = shuffled_X[:,i * mini_batch_size : (i + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,i * mini_batch_size : (i + 1) * mini_batch_size]

        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,miniBatchSizeNum * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,miniBatchSizeNum * mini_batch_size:]
        
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


#define the momentum algrithm
"""
此处使用到指数加权平均数，将其运用到深度神经网络的参数更新上
使用和指数加权平均相同结构的算法来运用到dw和db参数上
v = 0
for i in range(theta):
    v := beita*v + (1-beita)*theta[i] / (1 - beita^i)

以上是指数加权平均数的算法，我们可以将其应用到深度学习的参数更新中
v_dw0 = 0
db0 = 0
for i in range(0,layers)
    v_dw[i] += beita*v_dw[i - 1] + (1 - beita)*dw[i]
    v_dw[i] += beita*v_dw[i - 1] + (1 - beita)*dw[i]
"""
def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v


#the use the momentum algorithm to update the parameters
def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        #计算加速度v using the initialize v
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]

        #更新参数 using the original parameters and accelete v
        parameters["W" + str(l + 1)] += -learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] += -learning_rate * v["db" + str(l + 1)]
    return parameters,v



#define the adam optimize algorithm,the combine about momentum and PMSProp
def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v,s

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate = 0.01,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8):
    
    L = len(parameters) // 2
    v_corrected = {} #偏差修正后的值
    s_corrected = {}
    
    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1,t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1,t))

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])
        
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2,t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2,t))

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (v_corrected["db" + str(l + 1)] / np.sqrt(v_corrected["db" + str(l + 1)] + epsilon))


    return parameters,v,s

#define the model can run on the different optimize algorithm
def model(X,Y,layers_dims,optimizer,learning_rate = 0.0007,mini_batch_size = 64,beta = 0.9,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8,num_epochs = 10000,print_cost = True,is_plot = True):
    L = len(layers_dims)
    costs = []
    t = 0#t
    seed = 10

    parameters = opt_utils.initialize_parameters(layers_dims)
    if optimizer == "gd":
        pass#不适用任何优化器，直接使用梯度下降法
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)#使用动量
    elif optimizer == "adam":
        v,s = initialize_adam(parameters)#使用adam优化
    else:
        print("optimizer输入错误，程序退出！")
        exit(1)

    for i in range(num_epochs):
        
        seed = seed + 1
        minibatches = random_mini_batches(X,Y,mini_batch_size,seed)

        for minibatch in minibatches:
            #select one minibatch
            minibatch_X,minibatch_Y = minibatch
            #forward
            A3,cache = opt_utils.forward_propagation(minibatch_X,parameters)
            #cost
            cost = opt_utils.compute_cost(A3,minibatch_Y)
            #backward
            grads = opt_utils.backward_propagation(minibatch_X,minibatch_Y,cache)

            #update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters,grads,learning_rate)
            elif optimizer == "momentum":
                parameters,v = update_parameters_with_momentum(parameters,grads,v,beta,learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters,v,s = update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
        if i % 100 == 0:
            costs.append(cost)
            if print_cost and i % 1000 == 0:
                print("the" + str(i) + "th iteration,the cost is:" + str(cost))
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs(per 100)')
        plt.title("learning rate = " + str(learning_rate))
        plt.show()
    

    return parameters
                


    





























if __name__ == "__main__":
    

    """
    parameters,grads,learning_rate = testCase.update_parameters_with_gd_test_case()
    parameters = update_parameters_with_gd(parameters,grads,learning_rate)
    print(parameters)
    """
    
    """
    X,Y,mini_batch_size = testCase.random_mini_batches_test_case()
    mini_batches = random_mini_batches(X,Y,mini_batch_size)
    for i in range(len(mini_batches)):
        print('X: ',mini_batches[i][0].shape,'\n\ty: ',mini_batches[i][1].shape)
    """

    """
    parameters = testCase.initialize_velocity_test_case()
    v = initialize_velocity(parameters)
    print(v)
    """
    """
    parameters,grads,v = testCase.update_parameters_with_momentum_test_case()
    parameters,v = update_parameters_with_momentum(parameters,grads,v,beta = 0.9,learning_rate = 0.01)
    print(parameters,'\n',v)
    """
    """
    parameters = testCase.initialize_adam_test_case()
    v,s = initialize_adam(parameters)
    print(v,"\n",s)
    """
    """
    parameters,grads,v,s = testCase.update_parameters_with_adam_test_case()
    parameters,v,s = update_parameters_with_adam(parameters,grads,v,s,t = 2)
    print(parameters,'\n',v,'\n',s)
    """
    
    """
    test the gd
    train_X,train_Y = opt_utils.load_dataset(is_plot = True)
    layers_dims = [train_X.shape[0],5,2,1]
    parameters = model(train_X,train_Y,layers_dims,optimizer = "gd",is_plot = True)
    predictions = opt_utils.predict(train_X,train_Y,parameters)

    cw.plot_boundary('random',train_X,train_Y,parameters,left = -2.5,right = 2.5)
    """


    #test the momentum
    train_X,train_Y = opt_utils.load_dataset(is_plot = True)
    layers_dims = [train_X.shape[0],5,2,1]
    
    """
    parameters = model(train_X,train_Y,layers_dims,optimizer = "momentum",beta = 0.9,is_plot = True)
    predictions = opt_utils.predict(train_X,train_Y,parameters)
    
    """
    #test the adam
    parameters = model(train_X,train_Y,layers_dims,optimizer = "adam",beta = 0.9,is_plot = True)
    prediction = opt_utils.predict(train_X,train_Y,parameters)


    cw.plot_boundary('random',train_X,train_Y,parameters,left = -2.5,right = 2.5)


