"""
@Created by Mao on 2024/6/4
@Author:mao
@Github:https://github.com/masterchange13?tab=projects
@Gitee:https://gitee.com/master_change13
 
@File: solve.py
@IDE: PyCharm
 
@Time: 2024/6/4 16:41
@Motto:不积跬步无以至千里，不积小流无以成江海，程序人生的精彩需要坚持不懈地积累！
@target: 大厂offer，高年薪

@@ written by GuangZhi Mao

@from:
@code target:
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy


# init value
def init_function():
    lr = 1e-3
    espi = 1e-3
    x0 = sympy.Matrix([[1], [1]])
    x = sympy.Matrix([[100], [100]])

    return lr, espi, x0, x


# function to use
def function():
    x, y = sympy.symbols('x y')
    # func = sympy.log( 1 + (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2 ) / 10
    func = sympy.log(1 + (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2) / 10
    return func


# get direction of loss
def get_dk(f, x0):
    symbols = f.free_symbols
    count = len(symbols)

    matrix = sympy.zeros(count, 1)

    # for i, symbol in enumerate(symbols):
    #     matrix[i, 0] = f.diff(symbol)
    #     matrix = matrix.subs({symbol: x0[i]})

    # 正确累积替换操作
    for i, symbol in enumerate(symbols):
        derivative = f.diff(symbol)

        matrix[i, 0] = derivative.subs([(sym, val) for sym, val in zip(symbols, x0)])

    return -matrix


def update_val(x0, lr, dk):
    for i, d in enumerate(dk):
        x0[i, 0] += lr * dk[i, 0]

    return x0


def make_graph(x, y):
    plt.plot(x, y)
    plt.savefig("./graph/result.png")
    plt.show()


# def get_lr(f, dk, x, p, alpha=1, rho=0.5, c=0.1):
#     x1 = x + alpha * p
#     while f.subs({x: x1[0], y: x1[1]}) > f.subs({x: x[0], y: x[1]}) + c * alpha * get_dk(f, x) @ p:
#         # while f(x + alpha * p) > f(x) + c * alpha * dk(x) @ p:
#         alpha = rho * alpha
#     return alpha


def get_lr(f, dk, x0, p, alpha=1, rho=0.5, c=0.1):
    # 计算梯度在x0处的值，这里直接使用传入的梯度dk，避免了错误的get_dk调用
    grad_at_x0 = dk  # 这里dk已经是负梯度，所以无需再做符号替换

    while f.subs([(x, x0[0]), (y, x0[1])]) + c * alpha * grad_at_x0.dot(p) > f.subs(
            [(x, x0[0] + alpha * p[0]), (y, x0[1] + alpha * p[1])]):
        alpha *= rho
        if alpha < 1e-12:  # 防止alpha过小导致无限循环
            raise ValueError("Learning rate becomes too small. Check the Wolfe conditions or the input parameters.")

    return alpha


def get_lr(f, dk, x0, p, alpha=1, rho=0.5, c=0.1):
    # 将f转换为可计算的lambda函数
    f_lambdified = sympy.utilities.lambdify((x, y), f, 'numpy')

    grad_at_x0 = dk  # 这里dk已经是负梯度，所以无需再做符号替换

    # 使用lambdify后的函数进行数值比较
    while f_lambdified(x0[0], x0[1]) + c * alpha * grad_at_x0.dot(p) > f_lambdified(x0[0] + alpha * p[0], x0[1] + alpha * p[1]):
        alpha *= rho
        if alpha < 1e-12:  # 防止alpha过小导致无限循环
            raise ValueError("Learning rate becomes too small. Check the Wolfe conditions or the input parameters.")

    return alpha


def update_hession(n, x, s, H, dk):
    x_next = x + s
    y = get_dk(x_next) - dk
    rho = 1 / (y.T @ s)
    H = (np.eye(n) - rho * s.reshape(-1, 1) @ y.reshape(1, -1)) @ H @ (
            np.eye(n) - rho * y.reshape(-1, 1) @ s.reshape(1, -1)) + rho * s.reshape(-1, 1) @ s.reshape(1, -1)


if __name__ == '__main__':

    x, y = sympy.symbols('x y')
    lr, espi, x0, x1 = init_function()

    f = function()
    n = len(f.free_symbols)
    print("x0 is {}".format(x0))
    print("n is {}".format(n))

    y0 = f.subs({x: x0[0], y: x0[1]})
    y = f.subs({x: x1[0], y: x1[1]})
    print("y0 is {}".format(y0))

    H = sympy.eye(n)

    data_graph = []
    value_graph = []

    while (abs(y - y0) > espi):
        dk = get_dk(f, x0)
        # print("dk is {}".format(dk))

        p = -H * dk
        print("p is {}".format(p))
        lr = get_lr(f, dk, x0, p)

        # update x
        x1 = update_val(x0, lr, p)

        # update hession
        H = update_hession(n, x, x1, H, dk)

        y0 = y
        x0 = x1
        # print("f is {}".format(f))
        y = f.subs({x: x1[0], y: x1[1]})
        y = y.subs({y: x1[1]})
        # print("y0 is {}".format(y0))
        # print("y is {}".format(y))
        print("ans is {}".format(abs(y - y0)))

        data_graph.append(x1)
        value_graph.append(y)

    print("x is {}", x1)
    print("y is {}", y)

    print("data_graph is {}".format(data_graph))
    print("value_graph is {}".format(value_graph))

    # make_graph(data_graph, value_graph)
