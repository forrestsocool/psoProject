#约束条件1： x_list[0] -x_list[1] >= 0
def constrain0(x_list):
    return (x_list[0] - x_list[1])
    #return 0

#约束条件2： x_list[2] -x_list[3] >= 0
def constrain1(x_list):
    return (x_list[2] - x_list[3])

constrain_list = [
    constrain0,
    constrain1
]