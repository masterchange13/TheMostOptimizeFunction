#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :solve_own.py
# @Time      :2024/5/31 下午5:29
# @Author    ：mao

import numpy as np
import sympy as sp
from function import f
import matplotlib.pyplot as plt

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


def dfp_newton(f, x, iters):
    """
    实现DFP拟牛顿算法
    :param f: 原函数
    :param x: 初始值
    :param iters: 遍历的最大迭代次数
    :return: 每一次迭代的x，y"""

    res_x, res_y = [], []

    a = 1  # 定义初始步长
    H = np.eye(2)  # 初始化正定矩阵
    G = hesse(f, x)  # 初始化Hesse矩阵

    epsilon = 1e-3  # 一阶导g的第二范式的最小值（阈值）
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

        res_x.append(x)
        res_y.append(f(x))

    return res_x, res_y


if __name__ == '__main__':
    # 定义符号

    x1, x2 = sp.symbols('x1 x2')

    # 定义表达式，注意确保对数内的表达式在所有定义域内非负

    expression = (0 + (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2)

    # 确保对数内的表达式非负，这里直接假设表达式值总是正的，实际应用中需要验证

    log_expression = sp.log(expression)

    # 目标函数，这里先不除以10，稍后可以对最终结果进行此操作

    f_symbolic = log_expression / 10

    # 如果需要对特定数值点进行计算，可以将符号表达式中的变量替换为具体值

    x_data = np.array([[2], [1]])

    # 将符号表达式转换为针对x1, x2给定值的函数并计算

    f_value_at_point = f_symbolic.subs({x1: x_data[0][0], x2: x_data[1][0]}).evalf()

    # print(dfp_newton(f_symbolic, x_data, 20))
    res_x, res_y = dfp_newton(f, x_data, 1000)
    plt.plot(res_x, res_y, 'r-')
    plt.show()

    print("Symbolic expression:", f_symbolic)

    print("Value at the given point:", f_value_at_point)

    # 注意：dfp_newton函数未在你的代码中定义，如果是自定义的新顿法实现，请确保它能正确处理从SymPy转换来的函数

    # 如果dfp_newton期望的是数值函数和初始猜测值，你需要确保适当地处理从符号到数值的转换