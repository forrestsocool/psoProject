import numpy as np
import math
#功能描述：一个循环n次的PSO算法完整过程，返回这次运行的最小与最大的平均适应度 , 以及在线性能与离线性能
#[Result , OnLine , OffLine , MinMaxMeanAdapt] = PsoProcess ( SwarmSize , ParticleSize , ParticleScope , InitFunc , StepFindFunc , AdaptFunc , IsStep , IsDraw , LoopCount , IsPlot )
#输入参数：SwarmSize :种群大小的个数
#输入参数：ParticleSize：一个粒子的维数
#输入参数：ParticleScope :一个粒子在运算中各维的范围 ；
#         ParticleScope格式:
#           3维粒子的ParticleScope格式:
#                                   [x1Min , x1Max
#                                    x2Min , x2Max
#                                    x3Min , x3Max]
#
#输入参数 :InitFunc:初始化粒子群函数
#输入参数 :StepFindFunc:单步更新速度 ，位置函数
#输入参数：AdaptFunc：适应度函数
#输入参数：IsStep：是否每次迭代暂停；IsStep＝ 0 ，不暂停，否则暂停。缺省不暂停
#输入参数：IsDraw：是否图形化迭代过程；IsDraw＝ 0 ，不图形化迭代过程，否则，图形化表示。缺省不图形化表示
#输入参数：LoopCount：迭代的次数；缺省迭代100次
#输入参数：IsPlot：控制是否绘制在线性能与离线性能的图形表示；IsPlot = 0 , 不显示；
#                 IsPlot = 1 ；显示图形结果。缺省IsPlot = 1
#
#返回值：Result为经过迭代后得到的最优解
#返回值：OnLine为在线性能的数据
#返回值：OffLine为离线性能的数据
#返回值：MinMaxMeanAdapt为本次完整迭代得到的最小与最大的平均适应度
#
#用法[Result , OnLine , OffLine , MinMaxMeanAdapt] = PsoProcess ( SwarmSize , ParticleSize , ParticleScope , InitFunc , StepFindFunc , AdaptFunc , IsStep , IsDraw , LoopCount , IsPlot );
#

IsPlot = True
IsStep = False
IsDraw = False
def PsoProcess(SwarmSize, ParticleSize, ParticleScope, InitFunc, StepFindFunc, AdaptFunc, LoopCount):
    #容错控制
    if(np.shape(np.asarray(ParticleSize)) != ()):
        print("输入的粒子的维数错误，是一个1行1列的数据.")
        exit(-1)
    if (np.shape(np.asarray(ParticleScope))[0] != ParticleSize or np.shape(np.asarray(ParticleScope))[1] != 2):
        print("输入的粒子取值范围错误")
        exit(-1)

    #初始化种群
    ParSwarm, OptSwarm = InitFunc ( SwarmSize , ParticleSize , ParticleScope , AdaptFunc )


    #每一步的平均适应度
    MeanAdapt = np.zeros(LoopCount)

    #开始更新算法调用
    for k in range(0, LoopCount):
        print('--------------------------------------------------------')
        TempStr = '第 {} 次迭代'.format(k)
        print(TempStr)

        #调用一步迭代的算法
        ParSwarm , OptSwarm = StepFindFunc (ParSwarm , OptSwarm , AdaptFunc , ParticleScope , 0.95 , 0.4 , LoopCount , k+1)

        XResult = OptSwarm [SwarmSize,0:ParticleSize]
        YResult = AdaptFunc(XResult)
        str = '第 {} 次迭代的最优目标函数值 {}'.format(k, YResult)
        print(str)

        if IsStep:
            input("按任意键继续下次迭代...")
        print('--------------------------------------------------------')

        #记录每一步的平均适应度
        MeanAdapt[k] = np.mean(ParSwarm[:, 2*ParticleSize])

    #记录本次迭代得到的最优结果
    XResult = OptSwarm[SwarmSize, 0:ParticleSize]
    YResult = AdaptFunc(XResult)
    return YResult, XResult
