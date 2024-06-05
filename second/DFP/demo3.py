"""
@Created by Mao on 2024/6/5
@Author:mao
@Github:https://github.com/masterchange13?tab=projects
@Gitee:https://gitee.com/master_change13
 
@File: demo3.py
@IDE: PyCharm
 
@Time: 2024/6/5 13:15
@Motto:不积跬步无以至千里，不积小流无以成江海，程序人生的精彩需要坚持不懈地积累！
@target: 大厂offer，高年薪

@@ written by GuangZhi Mao

@from:
@code target:
"""

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt


def jacobian(f, x):  # 雅可比矩阵，求一阶导数
    a, b = np.shape(x)
    x1, x2 = sp.symbols('x1 x2')
    x3 = [x1, x2]
    df = np.zeros((a, 1))  # Initialize with zeros
    for i in range(a):
        df[i, 0] = sp.diff(f, x3[i]).subs({x1: x[0][0], x2: x[1][0]})
    return df


def hesse(f, x):  # Hessian矩阵
    a, b = np.shape(x)
    x1, x2 = sp.symbols('x1 x2')
    x3 = [x1, x2]
    G = np.zeros((a, a))
    for i in range(a):
        for j in range(a):
            G[i, j] = sp.diff(f, x3[i], x3[j]).subs({x1: x[0][0], x2: x[1][0]})
    return G


def make_graph2(f):
    # Create mesh grid data
    x1, x2 = sp.symbols('x1 x2')
    x1_vals = np.linspace(-5, 5, 150)
    x2_vals = np.linspace(-5, 5, 150)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Substitute values into the function
    Z = np.zeros_like(X1, dtype=np.float64)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = f.subs({x1: X1[i, j], x2: X2[i, j]})

    # Plot the 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='plasma', edgecolor='none')

    # Set labels and color bar
    ax.set_xlabel('x1', fontsize=14)
    ax.set_ylabel('x2', fontsize=14)
    ax.set_zlabel('f(x1, x2)', fontsize=14)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.ax.set_ylabel('Z Value', rotation=-90, va="bottom", fontsize=12)
    plt.title('3D plot of the function', fontsize=16)
    plt.savefig("./graph/3d_graph.png")
    plt.show()


def dfp_newton(f, x, iters):
    """
    实现DFP拟牛顿算法
    :param f: 原函数
    :param x: 初始值
    :param iters: 遍历的最大迭代次数
    :return: 最终更新完毕的x值
    """
    a = 1  # 定义初始步长
    H = np.eye(2)  # 初始化正定矩阵
    G = hesse(f, x)  # 初始化Hesse矩阵
    epsilon = 1e-3  # 一阶导g的第二范式的最小值（阈值）

    for i in range(iters):
        g = jacobian(f, x)
        if np.linalg.norm(g) < epsilon:
            xbest = [round(a[0]) for a in x]
            break
        d = -np.dot(H, g)
        a = -(np.dot(g.T, d) / np.dot(d.T, np.dot(G, d)))
        x_new = x + a * d
        g_new = jacobian(f, x_new)
        y = g_new - g
        s = x_new - x
        H = H + np.dot(s, s.T) / np.dot(s.T, y) - np.dot(H, np.dot(y, np.dot(y.T, H))) / np.dot(y.T, np.dot(H, y))
        G = hesse(f, x_new)
        x = x_new
        print("loading")

    make_graph2(f)
    return xbest


# Define the symbols and function
x1, x2 = sp.symbols('x1 x2')
x = np.array([[2], [1]])
f = sp.log(1 + (1.5 - x1 + x1 * x2)**2 + (2.25 - x1 + x1 * x2**2)**2 + (2.625 - x1 + x1 * x2**3)**2) / 10
print(dfp_newton(f, x, 20))
