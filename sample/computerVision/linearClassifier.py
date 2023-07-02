#!C:/Users/80521/AppData/Local/Programs/Python/Python38 python
# -*- coding=utf8 -*-

'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-05-24 13:16:36
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-05-24 13:16:36
 * @Description: this file, we will define the  lineara classifier class what based on 
 the gradient descent function. notice it is different from the knn algorithm.
 it will use the derivation to make the minimize the loss function. the knn algorithm
 is to find the min k L1 or L2 distance.
***********************************************************************'''
import numpy as np




class LinearClassifier(object):
    def __init__(self) -> None:
        self.W = None
    

    '''
     * @Author: weiyutao
     * @Date: 2023-05-24 14:08:19
     * @Parameters: 
            X: a numpy array of shape (N, D) containing train data; N samples and D features.
            y: a numpy array of shape (N, ) containing training labels;
            learning_rate: the learning rate for optimizing.
            reg: regularization strength.
            num_iters: number of steps to take when optimizing.
            batch_size: number of training examples to use at each step.
            verbose: boolean, if true, print progress during optimization.
     * @Return: a list containing the value of the loss function at each training iteration. 
     * @Description: you should input the original data and this function will return a list
     that stored the loss value for each iteration.
     '''
    def train(
        self, 
        X, 
        y, 
        learning_rate = 1e-3, 
        reg = 1e-5, 
        num_iters = 100, 
        batch_size = 200, 
        verbose = False
    ):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1

        # init the original weight for the instancement linearClassifier.
        # how to calculate the dimension of W? 
        # it is based on the input data X, the dimension of X is (N, D)
        # then X @ W = (N, D) @ (D, num_classes) = (N, num_classes)
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)


        # define the loss varibale.
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # then, how to define the data for each batch. we have define the batch_Size
            # so we can use the choice function in numpy to do it.
            # notice replacement is true is more efficient than false.
            # this function will return one list indicies that number of batch size, 
            # we have stored it used mask variable.
            mask = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[mask]
            y_batch = y[mask]

            # we will define the loss function at last.
            # the different method will have different loss function. 
            # notice, the grad is the derivation of weight, so it has the same shape as W.
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # iterate the weight param based on the learning rate and grad what the foremer
            # learning rate is the drop step length and the last grad is the drop slope.
            # because we are falling, so you should use the negative of slope.
            self.W -= learning_rate * grad
            if verbose and it % 100 == 0:
                    print("iterator %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history
    


    '''
     * @Author: weiyutao
     * @Date: 2023-05-24 14:38:27
     * @Parameters: 
     * @Return: 
     * @Description: this function will predict the data used the trained successful weight.
     '''
    def predict(self, X):
        # init one container to store the predict value.
        # it should has the same as the y_label
        y_pred = np.zeros(X.shape[0])
        # notice, the dimension of X @ W is (N, num_classes), 
        # the row numbers is N, and the number of columns is num_classese.
        # each value for each row is the probability value for the classes index.
        # so you should find the index of the max probabilty value in the result.
        # and it is the predict class. stored them used y_pred we have defined
        # in the former code.
        y_pred = np.argmax(np.dot(X, self.W), axis=1)

        return y_pred




    '''
     * @Author: weiyutao
     * @Date: 2023-05-24 14:32:08
     * @Parameters: 
            X_batch: a numpy array of shape (batch_size, D) containing a minibatch
            sample data of N.
            y_batch: a numpy array of shape (batch_size, ) containing a minibatch 
            y labels of N.
            reg: regularization strength.
     * @Return: a tuple containing the loss value and gradient with respect to self.W
        what is an array of the same shape as W.
     * @Description: this function is dedicated to calculate the loss and derivative.
     '''
    def loss(self, X_batch, y_batch, reg):
        # but we have not define any code in this function.
        # because it is inside of the parent class LinearClassifier.
        # we will define the implement of this loss function
        # in other child class.
        pass


class LinearSVM(LinearClassifier):
     def loss(self, X_batch, y_batch, reg):
          return svm_loss_vectorized(self.W, X_batch, y_batch, reg)