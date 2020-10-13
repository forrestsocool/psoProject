import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#Griewan函数
def Griewank(x_list):
    cols = len(x_list)
    x_list_squred = np.multiply(np.array(x_list), np.array(x_list))
    x_list_squre_sum = sum(x_list_squred)
    y1 = 1 / 4000 * x_list_squre_sum
    y2 = 1
    for h in range(0, cols):
        y2 = y2 * math.cos(x_list[h] / math.sqrt(h+1))
    y = y1 - y2 + 1
    y = -y
    return y

def DrawGriewank():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.arange(-8, 8.1, 0.1)
    y = np.arange(-8, 8.1, 0.1)
    X, Y = np.meshgrid(x, y)  # 转换成二维的矩阵坐标
    row = np.shape(X)[0]
    col = row= np.shape(X)[1]
    Z = np.zeros((row, col))

    for l in range (0, col):
        for h in range (0, row):
            Z[h][l] = Griewank([X[h][l], Y[h][l]])

    print(Z)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

#DrawGriewank()