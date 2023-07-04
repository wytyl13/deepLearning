#!C:/Users/80521/AppData/Local/Programs/Python/Python38 python
# -*- coding=utf8 -*-
'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-04 14:49:39
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-04 14:49:39
 * @Description: this file we will learn the chapter 1 in deep learning with pytorch.
***********************************************************************'''
""" 
why deep learning? 
    computer vision is certainly one of the fields that have been most impacted by the
    advant of deep leaning.
    the need to classify or interpret the content of natural images existed.
    very large datasets
    new constructs such as convolutional layers were invented and could be run quickly on GPUs 
    with unprecedented accuracy.
    all of these factors combined with the internet giants' desire to understand pictures taken
    by millions of users with their mobile decices and managed on said giants' platforms.
    so it is the reason why deep learning can do many things.

in this chapter, we will learn three popular pretrained models, one can label an image according
to its content, one can fabricate a new image from a real image, and the last can describe
the content of an image using proper english sentences.
we will learn how to load and run these pretrained models in pytorch.
the first cnn model is AlexNet what is a rather small network, but in our case, 
it is perfect for taking a first peek at neural network that does something and
learning how to run a pretrained version of it on a new thing. we can simple see
the structure of AlexNet. we can show all the models in pytorch used models instance.
we can also use all the models even if model has not added in the torch, we can use
it, because the torch will download the model directly.
these instance variabale can be called like a function. taking as input one or more images
and producing an equal number of scores for each of the 1000 ImageNet classes.

fo course, the preprocessing for the input image is necessary. and pytorch has defined 
the preprocessing function in torchvision module. it is transforms.

"""
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

class Chapter1:
    def __init__(self) -> None:
        pass
    
    def basic(self):

        # we can print all the model in pytorch and 
        # create one pretrained model used models.
        """
        print(dir(models))
        alexnet = models.AlexNet()
        print(alexnet)
        """

        # we can create the resnet101 used models instance.
        resnet = models.resnet101(pretrained=True)
        # print(resnet)

        # then, we can create the preprocessing instance used transforms in torchvision.
        # notice these normalize paramters what involved mean and std, we have got them when
        # we trained the samples. of course, we should normalize the test dataset used
        # the same mean and std we have got them during training.
        # resize means we will resize the image from the original size to 256*256, 
        # generally, this size is a big size for the original image. so the size of 
        # original image usually has smaller size.
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225])
        ])

        # we can read one image used pillo tool what is coded used python.
        # notice, this tool will call the image view in your current computer.
        img = Image.open("../../data/images/mv1.jpg")
        # the shape of the image is 640*400
        print(img.size)
        img_pre = preprocess(img)
        batch_t = torch.unsqueeze(img_pre, 0)
        print(img_pre.shape)
        resnet.eval()
        out = resnet(batch_t)
        print(out.shape)
