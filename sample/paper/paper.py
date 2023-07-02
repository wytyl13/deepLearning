"""******************************************************************************************
# Copyright (C) 2023. IEucd Inc. All rights reserved.
# @Author: weiyutao
# @Date: 2023-02-06 11:37:38
# @Last Modified by: weiyutao
# @Last Modified time: 2023-02-06 11:37:38
# @Description: we will will handle the statistics part of the paper what is about agriculture
# green technical efficiency used social sciences method. the construct about this paper is as follow.
# title:
#     the region differences of xinjiang agriculture green technical efficiency and the affecting factors analysis.

# the construct of the papaer
# start.
# abstract.
# keyword.
# ... some introduction ...
# 1 research methods and data sources.
#     1.1 research methods.
#         1.1.1 the measure of agriculture green technical efficiency.
#             invloved the DEA theory and expression, the measurement of the carbon emissions, all
#             indicators you have selected to calculate your agriculture green technical efficiency.
#         1.1.2 the measurement of the region difference.
#             what indicator you have selected to measure the region difference. thayer index or the other?
#             the expression and a brief description for your selected indicator. the standard about your
#             division in region. or you can referece others standard proposed by others.
#         1.1.3 the empirical analysis of the affecting factors about the agriculture green technical.
#             the empirical model, expression and some description about your indicators in this model.
#             you should distinguish the core index and other indicators. explain and be explained variables.
#             and some special pretreatment for you indicators should be also descripted.

#     1.2 data sources.
#         involved the original data sources and some pretreatment. 
# 2 the results and analysis
#     2.1 the temporal changes for agriculture green technical efficiency. 
#         you should introduce how to calculate the efficiency, use software or code by yourself?
#         and do some description about the retionality of statistics. just like F statistic or significance test.
#         you should at least list the following:
#         indicators, just like, the agriculture green technical efficiency for each year of each region.
#         a simple statistics involved the mean agriculture green technical efficiency for each region in all year.
#         the difference between the mean agriculture green technical efficiency of current year and the origianl year.
#         the mean agriculture green technical efficiency of all mean in all year.
#         of course, you can also define the variance about the two dimension array. you can just image this array
#         is a picture, if you imshow this array, it can show some reaction features. just like the mean of the
#         array means the mean efficiency for the whole array. it means the mean efficiency for xinjiang province.
#         the variance means the contrast for the whole array. you can caluculate the variance for the whole array.
#         it can show some issue. the bigger variance means the great change for agriculture green technical efficiency.
        
#         you should show one table for the above statistics and do a brief description based on the table.
#     2.2 the spatial distribution about the agriculture green technical efficiency.
#         you should at lest visualize the analysis results. just like you can show the efficiency of original year
#         and the current year first, second, you can define the efficiency level, you should at least define five levels.
#         0.0-0.2, 0.2-0.3, 0.3-0.4, 0.4-0.5, 0.5-1. or you can use clustering algorithm. then you should 
#         imshow the heat map about efficiency of each region in xinjiang.
#         then,  you can do some brief description about the heat map. and do some analysis.
#     2.3 the region differences for agriculture green technical efficiency.
#         visualize the thayer index result and descriped based on it.
#         you should at least use two figure to show the thayer index results. a histogram, a line chart.
#     2.4 the affecting factors analysis about the agriculture green technical efficiency.
#         how to calculate the empirical model. stata or other software. and do some specific description
#         about the model result. you should distinguish the different indicators to descript the result.
#         you should at least add one table for the stata regression results. a three wire table.
# 3 conclusion and implications.
#     3.1 the main conclusion.
#         1) the main result about the temporal changes for agriculture green technical efficiency
#         2) the main result about the spatial distribution features of each region for the efficiency.
#         3) thayer index result about difference of the region.
#         4) the main indicators that the main affecting factors to the agriculture green technical efficiency.
#     3.2 implications.
#         the advice about agriculture green technical efficiency.
#         the advice about the region difference for the efficiency.
#         the advice about the affecting factor for the efficiency.

# references:
#     ... 
#     ...
# end.
******************************************************************************************"""
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = 'SimHei' 
class Imshow:
    def __init__(self, name) -> None:
        self.name = name
    def imshow_3_1(data):
        labelx = data["district"].tolist()
        print(labelx)
        labely1 = data["Area（10KHM2）"]
        labely2 = data["RiceArea（10KHM2）"]
        labely3 = data["output"]
        fig,ax1 = plt.subplots()
        ax1.bar(labelx, labely1, lw = 1, color = 'gray', label = '农作物种植面积')
        ax1.bar(labelx, labely2, bottom = labely1, lw = 1, color = 'black', label = '水稻种植面积')
        ax1.set_title(label = '各地州种植业生产概况',fontsize = 15)
        ax1.set_xlabel(xlabel = '地州',fontsize=15)
        ax1.set_ylabel(ylabel = '种植面积（单位：10KHM2）',fontsize=15)
        plt.xticks(fontsize = 10)
        # plt.xticks(rotation = 300)
        plt.yticks(fontsize = 15)
        # plt.tick_params(labelsize=20)
        plt.legend(loc = 'upper center',fontsize=15)
        ax2 = ax1.twinx()
        ax2.plot(labelx, labely3, lw = 2, color = 'black', label = '农业生产总值', marker = 'o')
        plt.legend(fontsize=15)
        plt.yticks(fontsize = 15) 
        plt.show()
    def imshow_3_2(data):
        marker = ['s','p','x','o','^','v','+','*','8','h','1', '2', '3', '4']
        labelx = data.columns.values.tolist()[4:]
        m = data.shape[0]
        fig,ax = plt.subplots()
        for i in range(m):
            labely = np.array(data.loc[i, :]).tolist()[4:]
            name = np.array(data.loc[i, :])[0]
            ax.plot(labelx, labely, lw = 2, color = 'black', label = name, marker = marker[i])
        ax.set_title(label = '各个生产环节碳排放量',fontsize = 15)
        ax.set_xlabel(xlabel = '碳排放环节',fontsize=15)
        ax.set_ylabel(ylabel = '碳排放当量（单位：10^4KgCe）',fontsize=15)
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 15)
        plt.legend(loc = 'upper center',fontsize=10)
        plt.show()
    def imshow_3_3(data):
        labelx = data["district"].tolist()
        labely1 = data["output"]
        labely2 = data["total"]
        fig,ax1 = plt.subplots()
        ax1.bar(labelx, labely1, lw = 1, color = 'gray', hatch = "*", label = '农业产值')
        ax1.set_title(label = '农业产值和碳排放总量', fontsize = 15)
        ax1.set_xlabel(xlabel = '地州',fontsize=15)
        ax1.set_ylabel(ylabel = '农业产值（单位：亿元）',fontsize=15)
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 15)
        plt.legend(loc = 'upper center',fontsize=15)
        ax2 = ax1.twinx()
        ax2.plot(labelx, labely2, lw = 2, color = 'black', label = '碳排放总量', marker = 'o')
        ax2.set_ylabel(ylabel = '碳排放总量（单位：10^4KgCe）',fontsize=15)
        plt.yticks(fontsize = 15) 
        plt.legend(fontsize=15)
        plt.show()
    def imshow_3_5(data):
        labelx = data["district"].tolist()
        labely1 = data["mean"]
        labely2 = data["efficient"]
        fig,ax1 = plt.subplots()
        ax1.bar(labelx, labely1, lw = 1, color = 'gray', label = '人均种植规模')
        ax1.set_title(label = '人均种植规模和效率', fontsize = 15)
        ax1.set_xlabel(xlabel = '地州',fontsize=15)
        ax1.set_ylabel(ylabel = '人均种植规模（HM2/人）',fontsize=15)
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 15)
        plt.legend(loc = 'upper center',fontsize=15)
        ax2 = ax1.twinx()
        ax2.plot(labelx, labely2, lw = 2, color = 'black', label = '效率', marker = 'o')
        ax2.set_ylabel(ylabel = '效率值',fontsize=15)
        plt.yticks(fontsize = 15) 
        plt.legend(fontsize=15)
        plt.show()


    def imshow_3_6(data):
        labelx = data["时间"]
        ind = np.arange(len(labelx))
        width = 0.35
        labely1 = data["区域间泰尔指数"]
        labely2 = data["区域内泰尔指数"]
        labely3 = data["区域泰尔指数"]
        fig,ax1 = plt.subplots()
        ax1.bar([i for i in range(len(labelx))], labely1, width, lw = 1, color = 'white', edgecolor='black', label = "区域间泰尔指数")
        ax1.bar([i + width for i in range(len(labelx))], labely2, width, lw = 1, color = 'white', edgecolor='black', hatch = "xx", label = "区域内泰尔指数")
        ax1.plot(ind + width / 2, labely3, lw = 1, color = 'black', marker = "s", label = "区域泰尔指数")
        ax1.set_xticks(ind + width / 2)
        ax1.set_xticklabels(labelx)
        ax1.set_xlabel(xlabel = '年份',fontsize=20)
        ax1.set_ylabel(ylabel = '泰尔指数',fontsize=20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.legend(loc = 'upper left',fontsize=20)
        plt.show()
    

    def imshow_3_7(data):
        labelx = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
        ind = np.arange(len(labelx))
        mask1 = (data["区域"] == "东疆")
        mask2 = (data["区域"] == "南疆")
        mask3 = (data["区域"] == "北疆")
        labely1 = data.loc[mask1, "区域内单个泰尔指数"]
        labely2 = data.loc[mask2, "区域内单个泰尔指数"]
        labely3 = data.loc[mask3, "区域内单个泰尔指数"]
        fig,ax1 = plt.subplots()
        ax1.plot(ind, labely1, lw = 2, color = 'black', marker = "s", label = "东疆")
        ax1.plot(ind, labely2, lw = 2, color = 'black', marker = "D", label = "南疆")
        ax1.plot(ind, labely3, lw = 2, color = 'black', marker = "o", label = "北疆")
        ax1.set_xticks(ind)
        ax1.set_xticklabels(labelx)
        ax1.set_xlabel(xlabel = '年份',fontsize=20)
        ax1.set_ylabel(ylabel = '泰尔指数',fontsize=20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.legend(loc = 'upper left',fontsize=20)
        plt.show()