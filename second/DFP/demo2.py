"""
@Created by Mao on 2024/6/5
@Author:mao
@Github:https://github.com/masterchange13?tab=projects
@Gitee:https://gitee.com/master_change13
 
@File: demo2.py
@IDE: PyCharm
 
@Time: 2024/6/5 10:45
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
    a, b = np.shape(x)  # 判断变量维度
    x1, x2 = sp.symbols('x1 x2')  # 定义变量，如果多元的定义多元的
    x3 = [x1, x2]  # 将1变量放入列表中，方便查找和循环。有几个变量放几个
    df = np.array([[0.00000], [0.00000]])  # 定义一个空矩阵，将雅可比矩阵的值放入，保留多少位小数，小数点后面就有几个0。n元变量就加n个[]
    for i in range(a):  # 循环求值

        df[i, 0] = sp.diff(f, x3[i]).subs({x1: x[0][0], x2: x[1][0]})  # 求导和求值,n元的在subs后面补充

    return df


def hesse(f, x):  # hesse矩阵
    a, b = np.shape(x)
    x1, x2 = sp.symbols('x1 x2')
    x3 = [x1, x2]
    G = np.zeros((a, a))
    for i in range(a):
        for j in range(a):
            G[i, j] = sp.diff(f, x3[i], x3[j]).subs({x1: x[0][0], x2: x[1][0]})  # n元的在subs后面补充

    return G


def make_graph_norm(data_graph, value_graph):
    x1 = data_graph[:, 0]
    x2 = data_graph[:, 1]
    plt.plot(x2, value_graph, 'r-')
    plt.savefig('./graph/dfp_norm.png')
    plt.show()

def make_graph(data_graph, value_graph):
    x1 = data_graph[:, 0]
    x2 = data_graph[:, 1]
    x1, x2 = np.meshgrid(x1, x2)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, value_graph, cmap='rainbow')
    plt.savefig('./graph/dfp_newton.png')
    plt.show()

def make_graph2(f):
    # 创建网格数据，调整网格密度
    x1 = np.linspace(-5, 5, 150)  # 增加网格点数
    x2 = np.linspace(-5, 5, 150)
    X1, X2 = np.meshgrid(x1, x2)
    s1 = set()
    s2 = set()
    s1 = x1.tolist()
    s1 = set(s1)
    s2 = x2.tolist()
    s2 = set(s2)
    Z = f.subs({x1: s1, x2: s2})

    # 绘制3D图，调整颜色映射
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='plasma', edgecolor='none', linewidth=0, antialiased=False)  # 使用新的颜色映射

    # 设置坐标轴标签和图例
    ax.set_xlabel('x1', fontsize=14)
    ax.set_ylabel('x2', fontsize=14)
    ax.set_zlabel('f(x1, x2)', fontsize=14)
    ax.tick_params(labelsize=12)

    # 添加颜色条
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.ax.set_ylabel('Z Value', rotation=-90, va="bottom", fontsize=12)

    # 添加标题
    plt.title('3D plot of the function', fontsize=16)
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

    # 用于画图
    data_graph = []
    value_graph = []
    for i in range(1, iters):
        g = jacobian(f, x)

        if np.linalg.norm(g) < epsilon:
            xbest = []
            for a in x:
                xbest.append(round(a[0]))  # 将结果从矩阵中输出放到列表中并四舍五入
            break
        # 下面的迭代公式
        d = -np.dot(H, g)

        a = -(np.dot(g.T, d) / np.dot(d.T, np.dot(G, d)))

        # 更新x值
        x_new = x + a * d
        print("第 %d 次结果" % i)
        print(x_new)
        g_new = jacobian(f, x_new)
        y = g_new - g

        s = x_new - x
        # 更新H
        H = H + np.dot(s, s.T) / np.dot(s.T, y) - np.dot(H, np.dot(y, np.dot(y.T, H))) / np.dot(y.T, np.dot(H, y))
        # 更新G
        G = hesse(f, x_new)

        x = x_new

        data_graph.append(x)
        value_graph.append(f.subs({x1: x[0][0], x2: x[1][0]}))

    # data_graph and value_graph变成numpy格式
    data_graph = np.array(data_graph)
    value_graph = np.array(value_graph)

    # make_graph_norm(data_graph, value_graph)
    make_graph2(f)
    # make_graph(data_graph, value_graph)

    return xbest





x1, x2 = sp.symbols('x1 x2')  # 例子
x = np.array([[2], [1]])
# f = 2 * x1 ** 2 + x2 ** 2 - 4 * x1 + 2
f = sp.log(1 + (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2) / 10
print(dfp_newton(f, x, 20))
