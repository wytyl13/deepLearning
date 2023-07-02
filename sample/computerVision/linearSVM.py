'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-05-24 16:12:48
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-05-24 16:12:48
 * @Description: this file is dedicated to defining the function about linear svm algorithm.
***********************************************************************'''
import numpy as np


'''
 * @Author: weiyutao
 * @Date: 2023-05-24 17:39:06
 * @Parameters: 
 * @Return: 
 * @Description: notice, the loss function about svm algorithm is hinge loss.
 * Li = Σmax(0, sj - syi + δ) = L(y, W^T * xi + b) = max(0, 1-yi(W^T@xi+b))
 * Li is the loss value, sj is each score value we have calculated.
 * syi is the score that correct label correspond to. δ is one constant value.
 * we can define it used any number, just like we have used 1 in the follow case.
 * notice sj is each scores except the score that correct label correspond to.
 * reg can be defined used λ*ΣΣWkl^2
 * L = 1/N*ΣLi + λ*reg
 * because have the max function, so the original function hinge loss can not be differentialed.
 * so we should use subgradient.
 * dL/dW = -yixi if 1-yi(W^T@xi+b) > 0, otherwise 0
 * dL/db = -yi if 1-yi(W^T@xi+b) < 0, otherwise 0.
 * the svm dedicated to find the maximum separation hyperplane.
 * the support vector are all the vectors that smallest distance to the decision boundary.
 * the aims that decision boundary is to maximum the distance to the support vectors.
 '''
def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        # scores is the probabilty value list.
        scores = X[i].dot(W)
        # y[i] means the label for the current sample.
        # scores[y[i]] means the score that we have calculated for the correct label.
        # then notice, the score is not the max value. because W is begin from the initilize
        # value.
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            # except the score that correct label correspond to.
            # y[i] is the correct label.
            if j == y[i]:
                continue
            # scores[i] is each score except the score that correct label correspond to.
            # i is the δ value.
            margin = scores[j] - correct_class_score + 1

            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
    # loss is the sum value, we should calculate the average.
    loss /= num_train
    # add the regularization to the loss
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * 2 * W
    return loss, dW


'''
 * @Author: weiyutao
 * @Date: 2023-05-24 16:15:14
 * @Parameters: 
 * @Return: 
 * @Description: this function used vectorized method. it is more efficient than
 * using for loop to implement it.
 '''
def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0

    # initialize the gradient as zero.
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X, W)
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
    margin = np.maximum(0, scores - correct_class_score + 1)
    margin[np.arange(num_train), y] = 0
    loss = np.sum(margin) / num_train + reg * np.sum(W * W)
    margin[margin > 0] = 1
    correct_number = np.sum(margin, axis=1)
    margin[np.arange(num_train), y] -= correct_number
    dW = np.dot(X.T, margin) / num_train + reg * 2 * W

    return loss, dW
