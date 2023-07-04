#!C:/Users/80521/AppData/Local/Programs/Python/Python38 python
# -*- coding=utf8 -*-
'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-02 16:22:57
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-02 16:22:57
 * @Description: we will implement the yolo model in this file.
 *
***********************************************************************'''
""" 
some detection performance indicators.

the confusion matrix
            prediction
        positive    negative
    true    TP          FN
    false   FP          TN
T/F means the prediction of right and wrong.
P/N means the result of prediction.

precision = TP/(TP+FP), it can predict the accuracy of the prediction.
recall = TP/(TP+FP), it can predict the integerity.
accuracy = (TP+TN)/(TP+FN+FP+TN), the accuracy of all prediction.
F1 score = 2*precision*recall/(precision+recall)

iou = area of overlap / area of union: intersection over union 交并比
iou is equal to the intersection of two rectangle area / the union of two rectangle area.
just like we can set one threshold value based on the iou. it the iou is greater than
the threshold, classify the object detection as true positive(TP), else false positive(FP)

AP: average precision, the average of one class in one model: measure good or bad of one model for each class.
    AP, iou = 0.5
    AP, iou = 0.75
MAP: mean of average precision, the mean of all AP in one model: measure good or bad of one model for all class.

AP Across Scales:
    APsmall: small objects: area < 32^2
    APmedium: 32^2 < area < 96^2
    APlarge: area > 96^2
AR: average recall
    ARmax=1, given 1 detection per image.
    ARmax=10, given 10 detection per image.
    ARmax=100, given 100 detection per image.
AR Across Scales:
    ARsmall, area < 32^2
    ARmedium, 32^2 >area < 96^2
    ARlarge, 96^2 < area

then, we will learn how to train the dataset used exists yolo model.
simple to consider the problem. assume that we have five images as the train data.
and five images as the verify data.
then, we should mark the label first. we can use labelImg tool what is programed used python.
we can download it used pip directly and run it in terminal directly. 
you just need to pip install labelImg in terminal and run labelImg in terminal to
start the tool.
then you can get the label file used this tool, just like xml and txt. the last is used for yolo model.
the xml file is generally and easy to understand. because the content in xml file have point all the coordinates
we have marked in original image. and the txt file is not generally, different model do not have the same parameters
meaning. so we should focuse on the meaning of the txt file for yolo.
you can find five parameters in each line in txt file.
0 0.26 0.85 0.09 0.04
the first parameter 0 means the detected object index. 0 means the first detected object, 1 means the second.
we should consider the last four parameters based on the left upper point what is the original point. it mean the original
point is the left upper point of the image.
the second parameter 0.26 means the distance is 26% of the image width from the center point of the detected object 
to the original point.
the third parameter 0.85 means the disatnce is 85% of the image height from t he center point of the detected
object to the original point.
the fourth parameter 0.09 means the width of the detected object is 9% of the width of the image.
the fifth parameter 0.04 means the height of the detected object is 4% of the height of the image.
so we can get the location of the each detected object in one image. we should understande the meaning of each parameters
in the txt file.

then, you can store these images and label files into your yolo project and train them. of course, you
should store both the train data and verify data to get the train weight.
then, you can use the train weight file to predict the image.
the train program we have code in train.py and the predict program we have code in predict.py.
then, let's start to train our data and predict it.
we should consider three main train parameters. batch-size, iteration and epoch.
batch-size, the number of the sample each read by the memory.
iteration, train the batch-size number sample one time.
epoch, train all the samples one time.
just like the number of all the sample is 100.
batch-size = 10.
then, iteration = train 10 sample one time.
epoch = train all the sample one time.
mode detail:
just like the big data era, we have a large number of samples. so we can not train all
the sample just read one time by memory. so we should set the mini number samples.
so it is the meaning of batch-size parameter. and one read should complete at least two
steps what involved forward and backward propagation. so if we have completed these two
steps for one batch, we have iterated one time, so we have completed one iteration.
and we should read many times for all the samples. so if we have read all samples and 
completed all iterations, it means we have completed one epoch.
and we should consider why we should run many epoch? we should understand we have used
gradient descent algorithm to optimize our model. so we should set the epoch numbers
for each sample in order to we can minimize the loss function for the neural network model.
so it is the meaning for the epoch.
and the epoch, batch-size are all the super parameters.

load image.
you should reshape the image when you loaded the image.
scale the image based on the zoom ratio of the long side. it means you should scale all the long and short
side based on the ratio of the long side.
if the shape of image is 810*1080, then you want to reshape to 640*640
then, you should scale all the long and short side based on the 1080/640 = 1.69.
it means you should use the 1.69 to scale the long and short side.
"""