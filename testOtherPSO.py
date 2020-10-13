from pyswarm import pso
from benchmark import Griewank as gw

def banana(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5

def con(x):
    x1 = x[0]
    x2 = x[1]
    return [-(x1 + 0.25)**2 + 0.75*x2]

lb = [-50, -50, -50, -50, -50, -50, -50, -50, -50, -50]
ub = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

xopt, fopt = pso(gw.Griewank, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
    swarmsize=100, omega=1, phip=2, phig=2, maxiter=4000, minstep=1e-18,
    minfunc=1e-18, debug=False)

print(xopt)
print(fopt)
t = [ 50.     ,    -49.05542257, -50.  ,        50.      ,    50., -46.68343806, -50.      ,   -50.        , -47.90741291, -50.]
print(gw.Griewank(t))