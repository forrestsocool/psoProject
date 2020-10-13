import numpy as np
# 功能描述：初始化粒子群，限定粒子群的位置以及速度在指定的范围内
# InitSwarm(SwarmSize,ParticleSize,ParticleScope,AdaptFunc)
#
# 输入参数：SwarmSize:种群大小的个数
# 输入参数：ParticleSize：一个粒子的维数
# 输入参数：ParticleScope:一个粒子在运算中各维的范围；
#          ParticleScope格式:
#            3维粒子的ParticleScope格式:
#                                    [x1Min,x1Max
#                                     x2Min,x2Max
#                                     x3Min,x3Max]
#
# 输入参数：AdaptFunc：适应度函数
#
# 输出：ParSwarm初始化的粒子群
# 输出：OptSwarm粒子群当前最优解与全局最优解
#
def InitSwarm(SwarmSize, ParticleSize, ParticleScope, AdaptFunc):
    #容错控制
    if(np.shape(np.asarray(SwarmSize)) != ()):
        print("输入的种群的维数错误，是一个1行1列的数据.")
        exit(-1)
    if(np.shape(np.asarray(ParticleSize)) != ()):
        print("输入的粒子的维数错误，是一个1行1列的数据.")
        exit(-1)
    if (np.shape(np.asarray(ParticleScope))[0] != ParticleSize or np.shape(np.asarray(ParticleScope))[1] != 2):
        print("输入的粒子取值范围错误")
        exit(-1)

    #初始化粒子群矩阵,全部设为[0-1]随机数
    ParSwarm = np.random.rand(SwarmSize, 2 * ParticleSize + 1)

    #对粒子群中位置，速度的范围进行调节
    for k in range (0, ParticleSize):
        #将第k列向量设置到Scope范围之间
        ParSwarm[:,k] = ParSwarm[:,k] * (ParticleScope[k, 1] - ParticleScope[k, 0]) + ParticleScope[k, 0]
        #调节速度, 使得速度与位置范围一致
        ParSwarm[:, ParticleSize + k] = ParSwarm[:, ParticleSize + k] * (ParticleScope[k, 1] - ParticleScope[k, 0]) + ParticleScope[k, 0]

    #对每个粒子计算其适应度函数的值
    for k in range (0, SwarmSize):
        #每一行为一个粒子群，最后一位存储适应度函数的返回值
        ParSwarm[k, 2*ParticleSize] = AdaptFunc(ParSwarm[k, 0:ParticleSize])

    #初始化粒子群最优解矩阵
    #粒子群最优解矩阵全部设为零
    OptSwarm = np.zeros([SwarmSize+1, ParticleSize])

    #寻找适应度函数值得最大解得位置（行数）
    list_par = ParSwarm[:, 2*ParticleSize].tolist()
    max_adapt_index = list_par.index(max(list_par))  # 最大值的行号索引

    # OptSwarm最后一行保存最优解
    OptSwarm[0:SwarmSize, 0:ParticleSize] = ParSwarm[0:SwarmSize, 0:ParticleSize].copy()
    OptSwarm[SwarmSize, :] = ParSwarm[max_adapt_index, 0:ParticleSize]

    return ParSwarm, OptSwarm