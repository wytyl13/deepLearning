# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/8/18 16:59:59
#   @File Name : concatAndStack.py
#   @Description : 
#
#*****************************************************************
import torch

t1 = torch.tensor([1,1,1])
t2 = t1.unsqueeze(dim = 0)
t3 = t1.unsqueeze(dim = 1)
print(t1.shape,t2.shape,t3.shape)#3  1,3   3,1
#注意以上都是在重塑一个张量，实际上张量并不会改变，只是形状进行了改变


t4 = torch.tensor([1,1,1])#3
t5 = torch.tensor([2,2,2])#3
t6 = torch.tensor([3,3,3])#3

t7 = torch.cat((t4,t5,t6),dim = 0)#3+3+3=9
print(t7.shape)#9,该拼接不改变张量的轴数，而是改变张量现有轴的轴长

t8 = torch.stack((t4,t5,t6),dim = 0)#dim=0意思是在第一个索引处插入一个轴
print(t8.shape)#堆叠改变张量的轴数，在原有的基础上增加了一个轴，从3->3*3
print(t8)

t9 = torch.stack((t4,t5,t6),dim = 1)#在索引为1处进行堆叠
print(t9)
#什么时候选择拼接什么时候选择堆叠，，如果我们想要增加一个批次的轴，那么就选择堆叠，如果已经存在批处理轴了，那就是简单的拼接。那么最难的问题在于假如我们已经有了一个4阶张量，包含了批次轴，那么我们想要加入三个独立的图像，如何加呢？先按照0索引将新图像进行跌价我们会得到一个含有批次轴长为3的4阶张量，然后我们将该4阶张量和之前的要往已经存在的四阶张量里面拼接即可。其实stack就是升高维度然后进行拼接





























