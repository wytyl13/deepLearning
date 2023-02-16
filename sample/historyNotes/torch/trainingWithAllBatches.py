# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/8/18 10:00:47
#   @File Name : trainingWithAllBatches.py
#   @Description :使用所有的批次进行想训练 
#
#*****************************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth = 120)
torch.set_grad_enabled(True)


#define the network class
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 6,kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6,out_channels = 12,kernel_size = 5)
        
        self.fc1 = nn.Linear(in_features = 12 * 4 * 4,out_features = 120)
        self.fc2 = nn.Linear(in_features = 120,out_features = 60)
        self.out = nn.Linear(in_features = 60,out_features = 10)
    
    def forward(self,t):
        #the first hidden conv layer
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t,kernel_size = 2,stride = 2)
        
        #the second hidden conv layer 
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t,kernel_size = 2,stride = 2)

        #the first hidden linear layer
        t = t.reshape(-1,12 * 4 * 4)
        t = F.relu(self.fc1(t))
        
        #the second hidden linear layer
        t = F.relu(self.fc2(t))

        #th out layer
        t = self.out(t)

        return t

#define the train_Set
train_set = torchvision.datasets.FashionMNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()])
)


#define the function about calculate the correctly number
def get_num_correct(preds,labels):
    return preds.argmax(dim = 1).eq(labels).sum().item()


#define the general function to get the all preds
def get_all_preds(model,loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images,labels = batch
        
        preds = model(images)
        #add the preds to all_preds,based on the dim = 0
        all_preds = torch.cat(
            (all_preds,preds),
            dim = 0
        )
    return all_preds



if __name__ == "__main__":
    
    #print the version
    print(torch.__version__)
    print(torchvision.__version__)

    #new the class network
    network = Network()
    
    #create the trainloader to load the data from train_set
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = 100)

    #create the optimizer based on the Adam and network class and learning rate,because the optimizer involved the parameters update using the learning rate
    optimizer = optim.Adam(network.parameters(),lr = 0.01)
    
    """
    #读取一个批次
    batch = next(iter(train_loader))
    images,labels = batch

    preds = network(images)
    loss = F.cross_entropy(preds,labels)
    num = get_num_correct(preds,labels)
    print("优化前的loss和num：",loss,'\t',num)

    loss.backward()
    optimizer.step()

    #after backward and optimizer.step,the parameter could learn have updated,we can use the same images to chech the loss
    preds = network(images)
    loss = F.cross_entropy(preds,labels)
    num = get_num_correct(preds,labels)
    print("优化后的loss和num：",loss,'\t',num)
    """

    """
    以下就是将一个完整的数据集的所有批次进行一次完整的优化的过程
    #上面我们进行了一个批次的训练，下来我们传入所有的批次
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        images,labels = batch
        
        #forward and compute the loss
        preds = network(images)
        loss = F.cross_entropy(preds,labels)
        

        #a optimize involved the backward and optimize algorithm
        optimizer.zero_grad()#before each iteration the batch,you should clear the grad
        loss.backward()
        optimizer.step()

        total_loss += loss.item()#sum the each loss
        total_correct += get_num_correct(preds,labels)
    
    print("epoch:",0,"\n","total_correct:",total_correct,"\n","loss:",total_loss)

    print("正确率：",total_correct / len(train_set))
    """


    #以上就是一次完整的数据训练，我们称其为一个周期，下面我们使用进行5个周期的训练，以进行参数的优化
    for epoch in range(5):
        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images,labels = batch
            
            preds = network(images)
            loss = F.cross_entropy(preds,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss
            total_correct += get_num_correct(preds,labels)

            #注意以上的步骤顺序，前向传播->计算损失->grad清空->后向传播->优化
            #注意每一次对批次的循环都会先计算损失，然后进行优化，我们可以将这整个过程成为一次完整的优化，也可以仅仅称后向传播和优化为一次完整的优化
        print("epoch:",epoch,'\n',"total_correct:",total_correct,"\n","loss:",total_loss,"\n","正确率：",total_correct / len(train_set))



    #下面是将所有的训练结果也就是预测值放在一个张量里面，这样的操作更加简单直观。并且我们使用了上个步骤中训练好的网络参数network
    #下面我们来测试下混淆矩阵，首先我们需要创建一个通用的模型，来获取所有的预测值，将所有的预测值放在一个张量中
    prediction_loader = torch.utils.data.DataLoader(train_set,batch_size = 10000)
    train_preds = get_all_preds(network,prediction_loader)

    print(train_preds.shape)

    #检查梯度追踪是否开启，我们可以自定义开启或者关闭,如果我们不需要训练我们就不需要将梯度跟踪打开
    print(train_preds.requires_grad)
    print(train_preds.grad)
    #此时这个梯度张量是没有任何值的，因为我们并没有进行反向传播，所以grad并没有值，因为grad就是保存的偏导数
    print(train_preds.grad_fn)#这个是张量图像，如果可以返回值那就是我们在进行梯度跟踪。但是我们不需要梯度跟踪。所以我们可以将其关闭，可以选择全局或者局部
    
    #上下文管理进行局部操作,创建关闭梯度跟踪的加载器，也就是在创建张量的时候就要选择不跟踪
    with torch.no_grad():
        prediction_loader = torch.utils.data.DataLoader(train_set,batch_size = 10000)
        train_preds = get_all_preds(network,prediction_loader)#使用训练好的网络参数进行前向传播以求得预测值
    
    print(train_preds.requires_grad)#可以发现为梯度跟踪的状态false,这里是检查张量的梯度跟踪属性
    print(train_preds.grad_fn)#此时的返回值为none，所以梯度跟踪已经关闭。
    print(train_preds.grad)#检查梯度,不会有任何东西，因为我们没有做任何反向传播
    preds_correct = get_num_correct(train_preds,train_set.targets)#检查梯度函数图，none，我们没有跟踪梯度，所以我们使用了更少的内存。因为没有跟踪这个图
    print("total correct:",preds_correct)
    print("accuracy",preds_correct / len(train_set))

    #关闭梯度跟踪我们还可以使用对函数做注解的方式进行关闭，比如在get_all_preds函数定义的时候加上注解@torch.no_grad()
    
    #我们现在有了一个预测张量和一个标签张量，对应的维度分别为60000*1，注意，预测张量是一个60000*10的张量，我们需要使用argmax方法获取最大值对应索引
    #train_preds.argmax(dim = 1)#取出每一行的最大值对应的索引
    print(train_preds.argmax(dim = 1).shape,train_set.targets.shape)
    
    #我们可以使用stack方法将两个张量合并
    stacked = torch.stack(
        (train_set.targets,train_preds.argmax(dim = 1)),
        dim = 1
    )
    print(stacked.shape)#60000*2

    print(stacked[0].tolist())
    

    """
    #下面我们开始基于这样的数据组合来构建我们的混淆矩阵
    cmt = torch.zeros(10,10,dtype = torch.int32)

    print(cmt)

    for p in stacked:
        j,k = p.tolist()
        cmt[j,k] = cmt[j,k] + 1

    print(cmt)
    """

    #以上是自己创建的混淆矩阵，我们可使用sklearn创建
    from sklearn.metrics import confusion_matrix
    from resources.plotcm import plot_confusion_matrix

    cm = confusion_matrix(train_set.targets,train_preds.argmax(dim = 1))
    print(cm)



    names = ('T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')
    plt.figure(figsize = (10,10))
    plot_confusion_matrix(cm,names)







