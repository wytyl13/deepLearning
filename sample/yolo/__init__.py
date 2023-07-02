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
"""




