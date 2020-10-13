import numpy as np
import math

#功能描述：全局版本：基本的粒子群算法的单步更新位置 , 速度的算法
#
# BaseStepPso ( ParSwarm , OptSwarm , AdaptFunc , ParticleScope , MaxW , MinW , LoopCount , CurCount )
#
#输入参数：ParSwarm :粒子群矩阵 ，包含粒子的位置，速度与当前的目标函数值
#输入参数：OptSwarm：包含粒子群个体最优解与全局最优解的矩阵
#输入参数：ParticleScope :一个粒子在运算中各维的范围 ；
#输入参数：AdaptFunc：适应度函数
#输入参数：LoopCount：迭代的总次数
#输入参数：CurCount：当前迭代的次数
#
#返回值：含意同输入的同名参数
#
#用法：[ParSwarm , OptSwarm] = BaseStepPso ( ParSwarm , OptSwarm , AdaptFunc , ParticleScope , MaxW , MinW , LoopCount , CurCount )
#
def BaseStepPso ( ParSwarm , OptSwarm , AdaptFunc , ParticleScope , MaxW , MinW , LoopCount , CurCount ):

    #######   更改下面的代码，可以更改惯性因子   #######
    w = 1
    #线性递减策略
    if float(CurCount) / LoopCount > 0.85:
        w = MaxW - CurCount* ((MaxW-MinW) / LoopCount)
    #固定不变策略
    # w = 0.7
    #非线性递减策略，以凹函数递减1
    # w = (MaxW - MinW) * math.pow((CurCount / LoopCount) , 2) + (MinW - MaxW) * (2 * CurCount / LoopCount) + MaxW
    # 非线性递减策略，以凹函数递减2
    # w = MinW * math.pow((MaxW / MinW) , (1 / (1 + 10 * CurCount / LoopCount)))
    #######   更改上面的代码，可以更改惯性因子   #######

    #得到粒子群群体大小以及粒子维数的信息
    ParRow = np.shape(ParSwarm)[0]
    ParCol = int((np.shape(ParSwarm)[1] -1) / 2)
    #OptSwam和ParSwarm左上角相交部分求差值: pbest矩阵
    SubTract1 = OptSwarm[0:ParRow, :] - ParSwarm[:, 0:ParCol]

    #######   更改下面的代码，可以更改c1, c2的变化   #######
    c1 = 2
    c2 = 2
    #--------------------
    #con = 1
    #c1 = 4 - math.exp(-con * abs(np.mean(ParSwarm[:, 2 * ParCol])-AdaptFunc(OptSwarm[ParRow+1, :])))
    #c2 = 4 - c1
    #--------------------
    #######   更改上面的代码，可以更改c1, c2的变化   #######

    for row in range(0, ParRow):
        #全局最优减去当前解得到gbest
        SubTract2 = OptSwarm[ParRow,:] - ParSwarm[row, 0:ParCol]
        TempV = w * ParSwarm[row, ParCol:2*ParCol] + 2 * np.random.uniform(0, 1) * SubTract1[row, :] + 2 * np.random.uniform(0, 1) * SubTract2
        #限制速度的代码
        for h in range(0, int(ParCol)):
            if TempV[h] > ParticleScope[h, 1]:
                TempV[h] = ParticleScope[h, 1].copy()
            if TempV[h] < -ParticleScope[h, 1]:
                TempV[h] = -ParticleScope[h, 1].copy() + 1e-10
        #更新速度
        ParSwarm[row, ParCol:2*ParCol] = TempV.copy()

        #######   更改下面的代码，可以更改约束因子   #######
        # a = 1
        a = 0.729
        #######   更改上面的代码，可以更改约束因子   #######

        #更新每个粒子的位置
        TempPos = ParSwarm[row, 0:ParCol] + a*TempV
        for h in range(0, int(ParCol)):
            if TempPos[h] > ParticleScope[h, 1]:
                TempPos[h] = ParticleScope[h, 1].copy()
            if TempPos[h] <= ParticleScope[h, 0]:
                TempPos[h] = ParticleScope[h, 0] + 1e-10
        ParSwarm[row, 0:ParCol] = TempPos.copy()

        #计算每个粒子的新的适应度值，并且更新自身的历史最优解
        ParSwarm[row, 2*ParCol] = AdaptFunc(ParSwarm[row, 0:ParCol])
        if ParSwarm[row, 2*ParCol] > AdaptFunc(OptSwarm[row, 0:ParCol]):
                   OptSwarm[row, 0:ParCol] = ParSwarm[row, 0:ParCol]

        #寻找适应度函数值最大的解在矩阵中的位置（行数），进行全局最优的改变
        # 寻找适应度函数值得最大解得位置（行数）
        list_par = ParSwarm[:, 2 * ParCol].tolist()
        max_adapt_index = list_par.index(max(list_par))  # 最大值的行号索引
        if AdaptFunc(ParSwarm[max_adapt_index, 0:ParCol]) > AdaptFunc(OptSwarm[ParRow,:]):
            OptSwarm[ParRow, :] = ParSwarm[row, 0:ParCol].copy()

    #返回ParSwarm
    return ParSwarm, OptSwarm