#!C:/Users/80521/AppData/Local/Programs/Python/Python38 python
# -*- coding=utf8 -*-

'''**********************************************************************
 # Copyright (C) 2023. IEucd Inc. All rights reserved.
 # @Author: weiyutao
 # @Date: 2023-04-29 10:48:39
 # @Last Modified by: weiyutao
 # @Last Modified time: 2023-04-29 10:48:39
 # @Description: the entry function about SBM model what an efficient for each
 # dmu.
***********************************************************************'''


from sample.paper.dea import SBM
import pandas as pd
from sample.paper.map import MAP
import numpy as np
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide',True)



if __name__ == "__main__":
    """ 
    大论文
    data = pd.read_excel('c:/users/80521/desktop/SBM.xlsx')
    result = SBM.run(input_variable = ["input1", "input2", "input3", "input4", "input5", "input6", "input7"],\
                desirable_output = ["o1", "o2"], undesirable_output = ["s1"], dmu = ["dmu"], data = data)
    super_result = SBM.superSBM(data.values[:, 2:9], data.values[:, 9:11], data.values[:, 11:12])
    result.insert(2, "SUPER_TE", super_result)
    result.to_excel("c:/users/80521/desktop/output.xlsx", index = False)
    
    name_map = {
    '阿克苏地区':'阿克苏','阿勒泰地区':'阿勒泰','巴音郭楞蒙古自治州':'巴州','博尔塔拉蒙古自治州':'博州','昌吉回族自治州':'昌吉','哈密市':'哈密','和田地区':'和田','喀什地区':'喀什','克拉玛依市':'克拉玛依','克孜勒苏柯尔克孜自治州':'克州','石河子市':'石河子','塔城地区':'塔城','吐鲁番市':'吐鲁番','乌鲁木齐市':'乌鲁木齐','伊犁哈萨克自治州':'伊犁','铁门关市':'铁门关','阿拉尔市':'阿拉尔','图木舒克市':'图木舒克','可克达拉市':'可克达拉','北屯市':'北屯','双河市':'双河','昆玉市':'昆玉','五家渠市':'五家渠'
    }
    arrayAttention = np.array(data.reindex(columns = ["district", "o1"]))
    attentionRegion = arrayAttention.tolist()
    # noAttentionRegion = [["石河子", 0], ["铁门关", 0], ["阿拉尔", 0], ["图木舒克", 0], ["可克达拉", 0], ["北屯", 0], ["双河", 0], ["昆玉", 0], ["五家渠", 0]]
    MAP.createCityMap(name_map, attentionRegion, noAttentionRegion) 
    """

    """
    小论文
    data = pd.read_excel('c:/users/80521/desktop/test.xlsx')
    # transform 0 number to nan
    data.replace(0, pd.np.nan, inplace=True)
    print(data)
    result = SBM.run(input_variable = ["input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8"],\
                desirable_output = ["o1", "o2"], undesirable_output = ["s1"], dmu = ["dmu"], data = data)

    super_result = SBM.superSBM(data.values[:, 2:10], data.values[:, 10:12], data.values[:, 12:13])
    result.insert(2, "SUPER_TE", super_result)
    print(result)
    result.to_excel("c:/users/80521/desktop/test_.xlsx", index = False)
    print("WHOAMI")
    """

    # 投入按照亩数和非农就业人口占比计算，规则是经济作物投入较粮食作物投入大
    # 产出按照亩数和非农就业人口占比计算，规则是经济作物投入产出比较粮食作物投入产出比大
    # 非期望产出按照投入要素算。这个应该是固定的按照碳排放量计算的。

    data = pd.read_excel('c:/users/80521/desktop/test.xlsx')
    data.replace(0, 1, inplace=True)
    result = SBM.run(input_variable = ["input1", "input2", "input3", "input4", "input5", "input6", "input7"],\
                desirable_output = ["o1"], undesirable_output = ["s1"], dmu = ["dmu"], data = data)

    super_result = SBM.superSBM(data.values[:, 2:9], data.values[:, 9:10], data.values[:, 10:11])
    result.insert(2, "SUPER_TE", super_result)
    result['FINAL_TE'] = result.apply(lambda row: row['SUPER_TE'] if row['TE'] == 1 else row['TE'], axis=1)
    result.to_excel("c:/users/80521/desktop/test_.xlsx", index = False)
    print(result)

    # 中介检验这块存在一个问题，即我们分别使用了传统的三步法和bootstrap随机抽样方法做了检验都失败了。
    # 为什么？
    """ 
    首先，中介的机理是这样的。包括中介变量和解释变量和控制变量和被解释变量
    比如我们假设M变量是X对Y影响的中介变量，即X导致Y的原因是M。
    首先要确定X会对Y产生影响，这个是讨论X对Y的影响机理的前提。
        即tobit Y X, ll(0) ul(1.5)
    然后考虑X对中介变量M的影响。如果X不会导致M，那么也不会存在通过M去影响Y的假设了。
        即tobit M X, ll(0) ul(1)
    最后是重点，考虑控制X变量后M变量对Y的影响，这里一定要控制X，因为如果不控制X，将无法确定M是否在X发生的情况下
    对Y的影响。比如Z也会对Y产生影响，那么如果你不去控制X，直接对M 和Y 还有控制变量去回归，你将无法确定
    是否是因为船员出海而导致M，然后导致的对Y的影响，因为有可能是船员出海引发Z而导致Z对Y的影响。
    但是反过来，如果你做了如下回归
    tobit Y X M ..., ll(0) ul(1.5)
    然后你发现X的系数显著，但是M的系数不显著，这个能说明什么问题呢？
    很显然当你控制了X以后M不显著了，这说明M和Y没有关系，则中介效应失败。或者也可以说明X和M之前存在相关关系。
    但是如果不相关的话第二个方程就不显著了。

    那么问题来了，最后的答案肯定是非农就业显著影响农业绿色生产效率。
    正向影响还是负向影响呢？
    如果是正向影响，理由是什么？非农就业提升农户的技术采纳率。降低生产环节的要素投入。
    如果是负向影响，理由是什么？非农就业降低农民的生产积极性。那么如何调节呢？中介变量
    非农就业会降低农民的生产积极性，通过降低经济作物种植比重，然后来影响农户的生产效率。
    解决办法就是提升经济作物比重，如何提升？引进新产品。

    如果维持正向影响，那么中介效应就是农作物种植面积标准差。
    如果是负向影响，那么中介效应就是粮食作物种植面积占比。

    注意在数据的生成规则方面，如果加入了多个控制变量，那么控制变量对被解释变量影响的参数一定不能大于核心解释变量
    即考虑的核心解释变量的权重一定是大于控制变量的。
    如果以上问题没有考虑好，那么核心解释变量将不会影响显著，所以需要定义好这个层次。
    即非农就业人口占比的权重一定很大。然后是省内就业权重，然后是中介变量的权重。
    所以对于连续的我们要关注的控制变量和离散的我们要关注的控制变量，我们可以分开定义。

    比如我们可以先定义好核心解释变量和中介变量。这两个暂时都是连续的，所以我们可以加入连续型的权重。
    然后在定义离散的控制变量，比如家庭健康程度、居住环境和省内就业比重。

    还有一点需要注意的就是：
    变量的权重需要有比较，比如核心解释变量和控制变量，核心解释变量的权重肯定大于控制变量。
    核心解释变量和中介变量的权重，不能相差太大，否则其中的一个变量将会不显著，因为中介检验的第三个方程中要用到。

    还有一个就是可以按照比例权重也可以按照排名权重。

    那么为了解决以上变量放在一块的显著性影响，我们可以分开对变量加权重，比如我们可以先考虑核心解释变量的权重，
    然后再考虑中介变量，最后考虑控制变量。

    当然虚拟变量的权重最好加了，就是在上述权重都定义好的基础上再分类加权重。

    核心解释变量的权重按照排名定义。这个最大。
    剩下的要比这个小，否则控制住了就会不显著。

    是否可以考虑综合权重，不可以，是否可以考虑权重的权重，不可以。
    所以最后的结果是以上都不行。
    所以权重需要分开考虑，不能在一块计算。即不能都根据种植面积去计算。


    解决办法
        首先考虑核心解释变量权重：使用核心解释变量的比重或者排名都可以。直接使用比重就行。
        首先我们考虑中介的情况。
            首先根据核心解释变量作为权重生成投入数据。当然核心解释变量天然影响中介变量。
            我们可以发现，核心解释变量对效率的影响系数是0.36
            核心解释变量对中介变量的影响系数是0.69
            然而当加入中介变量作为控制变量去考虑X对Y的影响的时候，中介变量不显著了，这说明中介变量不是X对Y产生影响根本原因
            也就是说X对Y的影响另有他人，那么如何修改呢？
        最后结论是如果是调节效应，那么需要将核心解释变量的权重和调节变量的权重在一块考虑
        即核心解释变量对被解释变量的影响下数是调节变量的权重。
        如果是其他的那就单独考虑，比如中介效应。

        中介变量好调了，但是调节变量不好调，因为调节变量会影响到核心解释变量的系数。
        所以定义的时候需要将核心解释变量的系数定义为调节变量的函数。

    注意核心解释变量一定要作为面积的参数，否则将会很大。
    加入控制变量以后不显著了。

    注意控制变量不显著的原因主要是因为选择样本分布不均匀，即非农就业情况分布不均或者样本数量太小。
    所以我们还是在全部样本的基础上进行回归最合适了。
    统计好全部样本以后，首先查看样本的分布情况是否符合分析的情况，首先分布必须均匀，不能说集中在某一处。
    其次，样本数量要足够。
    那么下一个任务就是统计权农户的家庭情况和基本经营情况。
    主要是要确定农户的非农就业的分布要均匀，其次统计的时候一定要注意非农就业和种植面积一定要和自己的理论预期相同。
    """