# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/7/24 10:58:23
#   @File Name : .\course2Week1_2.py
#   @Description :
#*****************************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sys
sys.path.append('../data/course2Week1/')
import init_utils
import reg_utils
import gc_utils
import course2Week1 as cw



#define the model,the lambd = 0 means the 
def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost = True,is_plot = True,lambd = 0,keep_prob = 1):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],20,3,1]
    
    #initialize the parameters
    parameters = reg_utils.initialize_parameters(layers_dims)
    
    #forwardPropagation
    for i in range(0,num_iterations):
        if keep_prob == 1:
            a3,cache = reg_utils.forward_propagation(X,parameters)
        elif keep_prob < 1:
            a3,cache = forward_propagation_with_dropout(X,parameters,keep_prob)
        else:
            print("the input of the parameters erro,process exit!")
            exit
    
        #notice:the L2 influence the costFunction rather than the forwardPropagation
        if lambd == 0:
            cost = reg_utils.compute_cost(a3,Y)
        else:
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)

    
        #backwardPropagation
        #notice:the follow example is the one of L2 and dropout,not the all.so we need list follow conclusion
        if(lambd == 0 and keep_prob == 1):
            #no L2,no dropout
            grads = reg_utils.backward_propagation(X,Y,cache)
            
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X,Y,cache,lambd)
        
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)
    
    
        #update parameters
        parameters = reg_utils.update_parameters(parameters,grads,learning_rate)
    
        if i % 1000 == 0:
            costs.append(cost)
            if(print_cost):
                print("the " + str(i) + "th iteration,cost is " + str(cost))


    if is_plot:
        plt.plot(costs)
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title('learning rate =  ' + str(learning_rate))
        plt.show()


    return parameters


#define the forwardPropagation
#base on the a3,Y,parameters(use the w1 w2 w3 to compute,lambd is the super parameter)
def compute_cost_with_regularization(A3,Y,parameters,lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    first_cost = reg_utils.compute_cost(A3,Y)
    second_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

    cost = first_cost + second_cost

    return cost


#beacause we have changed the costFucntion,so we should redefined the backwardPropagation function.notice this is beacuse the backwardPropagation about the parameter W will be different from the backwardPropagation that no L2.
def backward_propagation_with_regularization(X,Y,cache,lambd):
    #define the added L2 model backwardPropagation
    m = X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3) = cache
    
    #the dZ3 is equal to no L2
    dZ3 = A3 - Y
    #dW3 = dZ3 * ?Z3/?W3   ?means the partial differential
    dW3 = (1 / m) * np.dot(dZ3,A2.T) + (lambd * W3 / m)
    #db3 = dZ3 * 1
    db3 = (1 / m) * np.sum(dZ3,axis = 1,keepdims = True)
    #here,we should list the forwardPropagation expression
    
    """
    Z1 = W1 @ X + b1    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2   A2 = relu(Z2)
    Z3 = W3 @ A2 + b3   A3 = sigmoid(Z3)
    dZ1 = ?J / ?Z1 ....
    A = sigmoid(Z) = 1 / (1 + e^-Z)   A' = A(1 - A)
    A = relu(Z) = maxinum(0,x)  A' = 1
    A = tanh(Z) = (1 + e^-2x) / (1 - e^-2x)   A' = 1 - A^2
    
    J = -1 / m * (y * (1og(A3) + (1 - y) * (1og(1 - A3))) + (lambd / 2m) * Wl^2

    dA3 = ?J / ?A3 = (1 / m) * (- y / A3 + (1 - y) / (1 - A3)) + 0
    base on above we can conclude only the dW parameter will be influenced when we add the L2
    dZ3 = dA3 * ?A3 / ?Z3 = dA3 * sigmoid(A3)' = A3 - Y
    dW3 = dZ3 * A2 + lambda * W3 / m
    db3 = dZ3
    """
    
    
    #so we can based on above to calculate the backwardPropagation with the L2
    #dA2 = ?J / ?A2 = (?J / ?Z3) * (?Z3 / ?A2) = dZ3 * W3
    dA2 = np.dot(W3.T,dZ3)
    
    #dZ2 = ?J / ?Z2 = (?J / ?A2) * (?A2 / ?Z2) = dA2 * 1 = dA2,but need to add the condition A2 > 0
    dZ2 = np.multiply(dA2,np.int64(A2 > 0))
    #dW2 = ?J / ?W2 = (?J / ?Z2) * (?Z2 / ?W2) = dZ2 * A1 + ((lambd * W2) / m)
    dW2 = (1 / m) * np.dot(dZ2,A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2,axis = 1,keepdims = True)

    #dA1 = (?j / ?Z2) * (?Z2 / ?A1) = dZ2 * W2
    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1,X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1,axis = 1,keepdims = True)

    gradients = {
        "dZ3":dZ3,"dW3":dW3,"db3":db3,"dA2":dA2,
        "dZ2":dZ2,"dW2":dW2,"db2":db2,"dA1":dA1,
        "dZ1":dZ1,"dW1":dW1,"db1":db1
    
    }
    
    return gradients


#define the dropput
"""
notice:
    keep_prob:the super parameter,0-1,means the rate of maintain
    D:the random int from 0 to 1,if D > keep_prob,D = 1;else D = 0;
    A * D / keep_prob
    the dropout will inflence the forward_propagation and backwardPropagation
    the L2 is based on the Wl,but the dropout is based on the A(relu/tanh)

    L2:
        forwardPropagation;layers = 3;equal to the forwardPropagation without the L2
            W1 @ X + b1 = Z1;relu(Z1) = A1;W2 @ A1 + b2 = Z2;relu(Z2) = A2;W3 @ A2 + b3 = Z3;sigmoid(Z3) = A3
        cost:J = -(1 / m) * (y * log(A3) + (1 - y) * log(1 - A3)) + [lambda / (2m)] * Σ Wl^2
        backwardPropagation;only the W parameters is not equal to without L2
    
    dropout:
        forwardPropagation;is not equal to the L2 or other;only influence the Al
            W1 @ X + b1 = Z1;relu(Z1) = A1;A1 * D1 / keep_prob = A1;
            W2 @ A1 + b2 = Z2;relu(Z2) = A2;A2 * D2 / keep_prob = A2;
            W3 @ A2 + b3 = Z3;sigmoid

"""
def forward_propagation_with_dropout(X,parameters,keep_prob = 0.5):
    np.random.seed(1)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1,X) + b1
    A1 = reg_utils.relu(Z1)

    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
    A1 = A1 / keep_prob

    Z2 = np.dot(W2,A1) + b2
    A2 = reg_utils.relu(Z2)
    D2 = np.random.rand(A2.shape[0],A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3,A2) + b3
    A3 = reg_utils.sigmoid(Z3)
    
    cache = (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3) 

    return A3,cache


def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    m = X.shape[1]
    (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b1,Z3,A3,W3,b3) = cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3,A2.T)
    db3 = (1 / m) * np.sum(dZ3,axis = 1,keepdims = True)
    dA2 = np.dot(W3.T,dZ3)

    dA2 = dA2 * D2
    dA2 = dA2 / keep_prob
    
    dZ2 = np.multiply(dA2,np.int64(A2 > 0))
    dW2 = 1 / m * np.dot(dZ2,A1.T)
    db2 = 1 / m * np.sum(dZ2,axis = 1,keepdims = True)

    dA1 = np.dot(W2.T,dZ2)
    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob

    dZ1 = np.multiply(dA1,np.int64(A1 > 0))
    dW1 = 1 / m * np.dot(dZ1,X.T)
    db1 = 1 / m * np.sum(dZ1,axis = 1,keepdims = True)

    gradients = {
                "dZ3":dZ3,"dW3":dW3,"db3":db3,"dA2":dA2,
                "dZ2":dZ2,"dW2":dW2,"db2":db2,"dA1":dA1,
                "dZ1":dZ1,"dW1":dW1,"db1":db1
    }
    

    return gradients



if  __name__ == "__main__":
    
    #load the datasets
    train_X,train_Y,test_X,test_Y = reg_utils.load_2D_dataset(is_plot = True)
    
    """
    #test
    #without the regfularization
    parameters = model(train_X,train_Y,is_plot = True)
    
    prediction_train = reg_utils.predict(train_X,train_Y,parameters)
    prediction_test = reg_utils.predict(test_X,test_Y,parameters)
    
    cw.plot_boundary('random',train_X,train_Y,parameters,left = -0.6,right = 0.6)
    """
    
    """
    #test with the L2
    parameters = model(train_X,train_Y,lambd = 0.7,is_plot = True)
    predictions_train = reg_utils.predict(train_X,train_Y,parameters)    
    predictions_test = reg_utils.predict(test_X,test_Y,parameters)

    cw.plot_boundary('random',train_X,train_Y,parameters,left = -0.7,right = 0.7)
    """


    parameters = model(train_X,train_Y,keep_prob = 0.86,learning_rate = 0.3,is_plot = True)
    predictions_train = reg_utils.predict(train_X,train_Y,parameters)
    predictions_test = reg_utils.predict(test_X,test_Y,parameters)

    cw.plot_boundary('random',train_X,train_Y,parameters,left = -0.7,right = 0.7)


