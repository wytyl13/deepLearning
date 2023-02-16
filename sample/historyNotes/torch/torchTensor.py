# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/8/9 10:42:27
#   @File Name : torchTensor.py
#   @Description : torch tensor
#
#*****************************************************************

import torch
import numpy as np
"""
#张量的声明，并且使用类型的声明，注意新版的torch、】支持不同类型的数据的加减乘除计算
a = torch.tensor([1],dtype = torch.int)
b = torch.tensor([1],dtype = torch.float32)
print((a + b).dtype)#注意运算后的数据类型就复杂不就简单。比如整数加浮点数运算的结果就是浮点的数据类型
#device命令指定cpu分配张量数据的设备。张量之间的运算必须发生在同一设备上存在的张量之间，随着我们称为更高级的用户，我们通常会使用多个设备
torch.device('cuda:0')

torch.layout()#指定了张量在内存中的存储方式
#张量包含统一类型的数据dtype
#张量之间的张量计算取决于dtype和device
#将张量t传送到GPU上运行
t.cuda()
"""


"""
#使用数据创建张量
torch.Tensor(data)
torch.tensor(data)
torch.as_tensor(data)
torch.from_numpy(data)
"""



"""
#以上四种方法都接受某种形式的数据并给我们一个torch.Tensor类的实例，但是当有多重方法可以达到相同的结果时，事情开始变得有点混乱。
c = np.array([1,2,3])
#分别使用以上四种方法将该数据转换为张量,但是注意最后的问题是我们不能使用最后三种方法声明一个张量，应该是版本的问题，所以我们就选择第一种办法去声明一个张量
c1 = torch.Tensor(c)
#c2 = torch.tensor(c)
#c3 = torch.as_tensor(c)
#c4 = torch.from_numpy(c)



print(c,'\n',type(c))
print("c1 = %s"%c1)

#创建一个维度为n的单位矩阵
d = torch.eye(2)
print(d)

#创建一个指定形状的0矩阵
e = torch.zeros([3,5])
print(e)




#创建元素为1的矩阵
f = torch.ones([3,4])
print(f)


#创建一个随机元素指定维度的张量
g = torch.rand([3,4])
print(g)#3*4

print(g.reshape([1,12]))#1*12
print(g.reshape([1,12]).shape)#二维

print(g.reshape([1,12]).squeeze())#一维  挤压
print(g.reshape([1,12]).squeeze().shape)#12,一维


print(g.reshape([1,12]).squeeze().unsqueeze(dim = 0))#1,12解压
print(g.reshape([1,12]).squeeze().unsqueeze(dim = 0).shape)#二维 1*12


#利用squeeze挤压函数构建flatten函数，使得对应的张量从二维降为1维度
def flatten(t):
    t = t.reshape(1,-1)
    t = t.squeeze()
    return t

t = flatten(g)
print(t.shape)#返回的是一个一维张量
"""


"""
#concatenationg Tensors
t1 = torch.tensor([[1,2],[3,4]])#2*2
t2 = torch.tensor([[5,6],[7,8]])#2*2

t3 = torch.cat((t1,t2),dim = 0)#dim = 0意思是按照行合并，也就是纵向合并4*2

print(t3)#4*2

t4 = torch.cat((t1,t2),dim = 1)#dim = 1,意思是按照列合并，也就是横向合并
print(t4)#2*4
"""



"""
#构建一个符合卷积神经网络输入的张量
#首先来构建一个3*4*4维度的张量  分别表示批次大小、高度和宽度
t1 = torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])#注意看张量的阶数，n阶张量具有n个左括号，因为这里要创建一个2阶所以有2个左括号
t2 = torch.tensor([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])
t3 = torch.tensor([[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3]])
t4 = torch.tensor([[4,4,4,4],[4,4,4,4],[4,4,4,4],[4,4,4,4]])
#print(t1,'\n',t2,'\n',t3,'\n',t4,'\n')
#注意此处不是普通的纵向或者横向合并，增加一个维度进行合并，也就是将这三个张量组装在一块，而不是融入到一块，将3个2阶轴长分别是4*4的张量组装为3阶轴长分别为3*4*4的张量，不能使用cat合并的方法，需要使用stack的方法，也就是栈的方法，压入栈中
t5 = torch.stack((t1,t2,t3))
print(t5,'\n')
print(t5.shape)#3*4*4
#此时我们有一个rank-3也就是3阶张量t5，其中包含一批三张图像，每张图像的维度是4*4，那么距离真正的神经网络输入层我们还缺少一个维度作为颜色通道，在批次和高度之间添加一个颜色通道，如果没有该通道，我们只能展示灰度图像，我们可以使用重塑的方法添加颜色通道
t5 = t5.reshape([3,1,4,4])#注意这块[]不是特别的严格要求
print(t5,'\n',t5.shape)
#注意附加的长度轴1不会改变张量中的元素数量，这是因为当我们乘以1的时候，组件值的成绩不会改变，此时我们生成了标准的神经网络的图像，只不过我们的颜色通道的轴长是1，所以我们只能表示灰度图像，第一个轴长3表示的是我们有三张图像，最后两个轴长分别为4表示的是高度和宽度的描述。我们可以使用索引号访问任何想要访问的东西

print(t5[0])#访问第一章图像，第一个维度为图像的批次，访问的是第一个图像，输出第一个图像的所有信息，输出的是一个三界张量，轴长分别为1*4*4
print(t5[0][0])#访问第一章图像的第一个颜色通道，因为本案例的颜色通道的轴长仅为1，所以只能访问到0这块]i
#print(t5[0][1])如果强行访问索引为1的颜色通道，会报错.访问颜色通道对应的会返回一个2阶张量，轴长分为被4*4
print(t5[0][0][0])#访问第一章图像的第一个颜色通道中的第一个行元素
print(t5[0][0][0][0])#访问第一张图像的第一个颜色通道中的第一行第一列的元素



#下面我们开始展平操作，我们首先将真个t5弄平
print(t5.reshape(1,-1).shape)#2阶张量轴长分别是1*48#这样我们会得到一个二维的数组，维度为1*48，每张图像展平后是16个元素，总共三张图像就是48个元素，我们可以使用挤压的方法将第一个维度挤压掉使得二维数组被挤压成一维数组,也可以直接取索引0拿到一维数组，或者直接reshape(-1)即可完成重塑和挤压的操作
print(t5.reshape(1,-1)[0].shape)#一阶张量轴长是48
print(t5.reshape(-1).shape)#一阶张量轴长是48
print(t5.view(t5.numel()).shape)#一阶张量轴长是48
print(t5.flatten().shape)#一阶张量轴长是48


#但是我们想要的并不是全部展平，而是按照对应的维度展平，也就是说我们不想展平全部的图像，因为我们每个批次会有多个图像，我们指向按照每一张图像依次展开存放，那么我们可以使用flatten传递参数的方法实现这个功能
print(t5.flatten(start_dim = 1).shape)#这里的start_dim的意思是从哪个轴往后开始展平，也就是不战平第一个轴也就是批次,展平操作后会生成一个2阶张量，轴长分别为3*16，注意我们是从颜色通道开始往后的轴全部展平，前面的不做展平,前面有n个轴也就不对n个轴进行展平，那么展平后会生成n+1阶张量。也就是说对n阶张量从第m阶开始展平后，会生成一个m+1阶的张量，也就是说从第k阶展平会生成一个k+1阶的张量，注意这里考虑了索引从0开始的，第n阶在实际的程序中是n+1阶，如果换算成现实，那么如果想从第一阶开始展平，会生成一个一阶张量，如果想从第k阶开始展平，会生成一个K阶张量。
"""

t1 = torch.ones(1,2,2)
t2 = torch.ones(1,2,2) + 1
t3 = torch.ones(1,2,2) + 2
print(t1,'\n',t2,'\n',t3)
print(torch.cat((t1,t2,t3),dim = 0).shape)

t4 = torch.ones(2,2)
t5 = torch.ones(2,2) + 1
t6 = torch.ones(2,2) + 2

print(t4,'\n',t5,'\n',t6)
print(torch.stack((t4,t5,t6)).shape)#注意stack的用法，需要将各个张量使用括号括起来

print(np.broadcast_to(2,t1.shape))

print(t1.eq(t3),'\n',t1.ge(t3),'\n',t1.gt(t3),'\n',t1.lt(t3),'\n',t1.le(t3))#从前往后分别为= >= > < <=
print(t1.abs(),'\n',t1.sqrt(),'\n',t1.neg(),'\n',t1.neg().abs())#从前往后分别是取整 开根号 取负 取反并且取整
#注意元素操作 逐点操作 和组件操作都是指相同的意思

print(t4.argmax(dim = 0))

print(t4.mean().item())
print(t4.mean(dim = 0).tolist())
#print(t2.mean(dim = 0).numpy())












































