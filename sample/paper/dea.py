import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
import pandas as pd

class DEA:
    def __init__(self, name) -> None:
        self.name = name
    
    def getSamples(sampleNumbers, inputNumbers, expectOutputNumbers, undesiredOutputNumbers):
        X = np.random.randint(1, 10, (sampleNumbers, inputNumbers))
        Y = np.random.randint(1, 10, (sampleNumbers, expectOutputNumbers))
        Z = np.random.randint(1, 10, (sampleNumbers, undesiredOutputNumbers))
        return X, Y, Z
    
    def run(X, Y, Z):
        return 0


class SBM(DEA):
    def __init__(self, name) -> None:
        super().__init__(name)
    """
    parameters:
    input_varibale: 投入字段
    desirable_output: 期望产出字段列表
    undesirable_output: 非期望产出字段列表
    dmu: 决策单元
    data: 数据表
    method: calculate method
    return: [dmu, TE, slack...]
    """
    def run(input_variable, desirable_output, undesirable_output, dmu, data, method = 'revised simplex'):
        result = pd.DataFrame(columns = ['dmu', 'TE'], index = data.index)
        result['dmu'] = data[dmu]
        dmu_counts = data.shape[0]
        m = len(input_variable)
        s1 = len(desirable_output)
        s2 = len(undesirable_output)
        total = dmu_counts + m + s1 + s2 + 1
        cols = input_variable + desirable_output + undesirable_output
        newcols = []
        for j in cols:
            newcols.append(j + '_slack')
            result[j + '_slack'] = np.nan
        for i in range(dmu_counts):
            c = [0] * dmu_counts + [1] + list(-1 / (m * data.loc[i, input_variable])) + [0] * (s1 + s2)
            
            # define the contraint container
            A_eq = [[0] * dmu_counts + [1] + [0] * m + list(1 / ((s1 + s2) * data.loc[i, desirable_output]))
                    +   list(1 / ((s1 + s2) * data.loc[i, undesirable_output]))]

            # the first contraint
            for j1 in range(m):
                list1 = [0] * m
                list1[j1] = 1
                eq1 = list(data[input_variable[j1]]) + [-data.loc[i ,input_variable[j1]]] + list1 + [0] * (s1 + s2)
                A_eq.append(eq1)
            
            # the second contraint
            for j2 in range(s1):
                list2 = [0] * s1
                list2[j2] = -1
                eq2 = list(data[desirable_output[j2]]) + [-data.loc[i, desirable_output[j2]]] + [0] * m + list2 + [0] * s2
                A_eq.append(eq2)
            
            # the third contraint
            for j3 in range(s2):
                list3 = [0] * s2
                list3[j3] = 1
                eq3 = list(data[undesirable_output[j3]]) + [-data.loc[i, undesirable_output[j3]]] + [0] * (m + s1) + list3
                A_eq.append(eq3) 

            b_eq = [1] + [0] * (m + s1 + s2)
            bounds = [(0, None)] * total

            # calculate the result
            op1 = linprog(c = c, A_eq = A_eq, b_eq = b_eq, bounds = bounds, method = method)
            result.loc[i, 'TE'] = op1.fun
            result.loc[i, newcols] = op1.x[dmu_counts + 1 :]
        
        return result
    
    """
    input_variable: the input variable list.
    desirable_output: the desirable output variable list
    undesirable_output matrix: the undesirable output variable list.
    """
    def superSBM(input_variable, desirable_output, undesirable_output):
        x = input_variable.T
        y_g = desirable_output.T
        y_b = undesirable_output.reshape(1,-1)  # reshape的原因是只有一个非期望产出变量，上一行得到的y_b是一个一维的向量，需要将其转换成一个1×n的二维行向量，否则后续进行矩阵操作容易报错
        m, n = x.shape
        s1 = y_g.shape[0]
        s2 = y_b.shape[0]
        theta = []
        for i in range(n):
            f = np.concatenate([np.zeros(n), 1/(m * x[:, i]),
                                np.zeros(s1+s2), np.array([1])])

            Aeq = np.hstack([np.zeros(n),
                            np.zeros(m),
                            -1/((s1+s2)*(y_g[:, i])),
                            -1/((s1+s2)*(y_b[:, i])),
                            np.array([1])]).reshape(1, -1)
            beq = np.array([1])

            Aub1 = np.hstack([x,
                            -np.identity(m),
                            np.zeros((m, s1+s2)),
                            -x[:, i, None]])

            Aub2 = np.hstack([-y_g,
                            np.zeros((s1, m)),
                            -np.identity(s1),
                            np.zeros((s1, s2)),
                            y_g[:, i, None]])

            Aub3 = np.hstack([y_b,
                            np.zeros((s2, m)),
                            np.zeros((s2, s1)),
                            -np.identity(s2),
                            -y_b[:, i, None]])

            Aub = np.vstack([Aub1,Aub2,Aub3])
            Aub[:,i]=0
            bub = np.zeros(m+s1+s2)
            bounds = tuple([(0, None) for i in range(n+s1+s2+m+1)])
            res = linprog(c=f, A_ub=Aub, b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds)
            theta.append(res.fun)
        return theta