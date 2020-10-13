from benchmark import Griewank as gw
from benchmark import Conditions as con
import Constrains
import Bootstrap


currFunc = gw.Griewank

def myFitFunc(x_list):
    con_pen, Bootstrap.rho = Constrains.getConstrainedAppendValue(x_list, con.constrain_list, Bootstrap.alpha, Bootstrap.rho)
    #print("rho : {} \n".format(Bootstrap.rho))
    return currFunc(x_list) + con_pen