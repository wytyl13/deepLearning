#!C:/Users/80521/AppData/Local/Programs/Python/Python38 python
# -*- coding=utf8 -*-

'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-04-30 12:09:38
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-04-30 12:09:38
 * @Description: the stanford course. computer vision. cs231n
L1 distance: just like the L1 distance of two images.
it is the mean difference between the value of corresponding pixels.
the L1 distance is one value what represent the similar degree for these
two images. so you must calculate each test samples with the object image
what you want to predict. but how to train in knn algorithm? the train process
for knn algorithm is to input the samples in memory. so you can find the train process
for knn is very simple, but the test or the predict process is very complex, 
becuase we should calculate L1 distance or L2 distance about the object image with each smaple.

L2 distance is one value what is similar to the L1 distant, but the difference is 
L1 distance is the mean of the difference of corresponding pixels for the object image with each smaple.
L2 distance is the root of sum of square of two image.

L1: d1(I1, I2) = xigema|I1 - I2|
L2: d2(I1, I2) = xigema((I1 - I2)^2)^(1/2)
K is the super parameter what means the numbers of samples you will find the nearest distance in 
the object image and with each sample. the super parameter can influence the efficient of the knn 
algorithm. so we should select the best k parameter used cross validation.
once we found the k nearest sample for the object image. we should get the label that in most times
in these k samples. the label will be the predict label for the object image.
***********************************************************************'''
import numpy as np
import torchvision.datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


if __name__ == "__main__":
    train_data = torchvision.datasets.CIFAR10(root='data', train=True, transform=torchvision.transforms.ToTensor(), download=False)
    test_data = torchvision.datasets.CIFAR10(root='data', train=False, transform=torchvision.transforms.ToTensor(), download=False)
    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # you can use get each sample used enumerate what is like one iterotor in cpp.
    # each sample has two data structure what involved X and y label. X is all the feature pixel for one sample what
    # the dimension is dimension * m * n. so generally the dimension is one or three, m is equal to n.
    # and you should notice the pixel value is decided by the type of the current smaple image.
    # the range value will be from 0 to 255 if the image is 8bits. 2^8 = 256(0, 255)
    # m and n is the pixel numbers for one image. it can also be named the resolution ratio for one image.
    # so you can iterator the train_data and get the features and labels for each sample used input and target variable.
    X = np.array()
    for i, (input, target) in enumerate(train_data):
        # you can reshape the three dimension array to 1 dimension array.
        # but the data type is tensor. you should transform it if you want to use array or list.
        # generally, we used array. of course, you can use tensor directly if you used torch lib.
        # of course, you can use reshape, flatten or ravel. you can use tensor.numpy(), or
        # tensorflow.convert_to_tensor(numpy_variable)
        input_new = input.reshape(-1).reshape(1, -1)
        list.append(input_new)
    print(np.array(list).shape)
    # print(train_data.data[0].shape)
    # print the first image.
    """     
    fig = plt.figure()
    plt.imshow(train_data.data[0])
    plt.show() 
    """



