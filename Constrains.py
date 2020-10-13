import math

def getConstrainedAppendValue(x_list, constrain_list, alpha, rho):
    constrain_append = 0.0
    good_num = 0
    for constrain_func in constrain_list:
        constrain_i = constrain_func(x_list)
        if constrain_i >= 0:
            good_num += 1
        constrain_append += math.pow(10, alpha*(1-rho)) * min(0, constrain_i)
    rho = float(good_num) / len(constrain_list)
    return constrain_append, rho
