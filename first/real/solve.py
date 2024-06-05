"""
@Created by Mao on 2024/6/4
@Author:mao
@Github:https://github.com/masterchange13?tab=projects
@Gitee:https://gitee.com/master_change13
 
@File: solve.py
@IDE: PyCharm
 
@Time: 2024/6/4 16:52
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
    espi = 1e-5
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
    x = np.array(x)
    y = np.array(y)

    # 确保x和y可以适配，这里假设y的每个值对应x的一列
    # assert x.shape[1] == len(y), "x的列数应与y的长度相匹配"
    # 绘制图像
    plt.figure(figsize=(10, 6))
    for idx, column in enumerate(x.T):  # 使用转置来遍历每一列
        plt.plot(column, y[idx] * np.ones_like(column), label=f'Series {idx + 1}',
                 marker='o')  # 用y值乘以ones_like创建与x相同长度的序列用于绘图
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Multiple Series Plot')
    plt.legend()
    plt.grid(True)

    plt.savefig("./graph/result.png")
    plt.show()

def make_graph2(x, y):
    x = np.array(x)
    x = x[:, 0]
    x2 = x[:, 1]
    y = np.array(y)

    plt.figure()
    plt.plot(x, x2, y, '-r')
    plt.savefig("./graph/result.png")

    plt.show()

def make_graph_3d(x, y):
    x = np.array(x)
    x = x[:, 0]
    x2 = x[:, 1]
    y = np.array(y)

    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, y, cmap='coolwarm', alpha=0.8)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')

    x_list = np.array(x)
    ax.plot(x_list[:, 0], x_list[:, 1], f(x_list.T), color='black', linewidth=2)

    plt.savefig("./graph/result.png")
    plt.show()

if __name__ == '__main__':

    x, y = sympy.symbols('x y')
    lr, espi, x0, x1 = init_function()
    f = function()
    print("x0 is {}".format(x0))

    y0 = f.subs({x: x0[0], y: x0[1]})
    y = f.subs({x: x1[0], y: x1[1]})
    print("y0 is {}".format(y0))

    data_graph = []
    value_graph = []

    while (abs(y - y0) > espi):
        dk = get_dk(f, x0)
        # print("dk is {}".format(dk))

        x1 = update_val(x0, lr, dk)

        # print("x1 is {}".format(x1))

        y0 = y
        # print("f is {}".format(f))
        y = f.subs({x: x1[0], y: x1[1]})
        y = y.subs({y: x1[1]})
        # print("y0 is {}".format(y0))
        # print("y is {}".format(y))
        print("ans is {}".format(abs(y - y0)))

        data = [x1[0], x1[1]]
        data_graph.append(data)
        value_graph.append(y)

    print("x is {}", x1)
    print("y is {}", y)

    print("data_graph is {}".format(data_graph))
    # print("value_graph is {}".format(value_graph))

    # temp = np.array(data_graph)
    # print(temp[:, 0])

    make_graph2(data_graph, value_graph)