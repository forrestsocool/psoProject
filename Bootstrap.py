import numpy as np
import PsoProcess as pso
import InitSwarm as init
import BaseStepPso as bsp
from benchmark import FitFunc

alpha = 2.0
rho = 0.0

def main():
    dim = 10
    Scope = np.zeros((dim, 2))
    Scope[:, 0] = -500
    Scope[:, 1] = 500

    YResult, XResult = pso.PsoProcess(50, dim, Scope, init.InitSwarm, bsp.BaseStepPso, FitFunc.myFitFunc, 15000)
    print(XResult)
    print(YResult)

if __name__ == "__main__":
    main()