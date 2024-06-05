"""
@Created by Mao on 2024/6/5
@Author:mao
@Github:https://github.com/masterchange13?tab=projects
@Gitee:https://gitee.com/master_change13
 
@File: demo.py
@IDE: PyCharm
 
@Time: 2024/6/5 9:02
@Motto:不积跬步无以至千里，不积小流无以成江海，程序人生的精彩需要坚持不懈地积累！
@target: 大厂offer，高年薪

@@ written by GuangZhi Mao

@from:
@code target:
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x):
    # return x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[0] * x[1] - 4 * x[0] - 6 * x[1]
    return np.log(1 + (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2)


def grad(x):
    # return np.array([2 * x[0] + 2 * x[1] - 4, 4 * x[1] + 2 * x[0] - 6])
    return np.array([])


def dfp(func, grad, x0, max_iter=1000, tol=1e-6):
    n = len(x0)
    H = np.eye(n)
    x = x0.copy()
    x_list = [x]
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        p = -H @ g
        alpha = backtracking_line_search(func, grad, x, p)
        s = alpha * p
        x_next = x + s
        y = grad(x_next) - g
        rho = 1 / (y.T @ s)
        H = (np.eye(n) - rho * s.reshape(-1, 1) @ y.reshape(1, -1)) @ H @ (
                    np.eye(n) - rho * y.reshape(-1, 1) @ s.reshape(1, -1)) + rho * s.reshape(-1, 1) @ s.reshape(1, -1)
        x = x_next
        x_list.append(x)
    return x_list


def backtracking_line_search(func, grad, x, p, alpha=1, rho=0.5, c=0.1):
    while func(x + alpha * p) > func(x) + c * alpha * grad(x) @ p:
        alpha = rho * alpha
    return alpha


x0 = np.array([0, 0])
x_list = dfp(f, grad, x0)

x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f([X1, X2])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='coolwarm', alpha=0.8)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

x_list = np.array(x_list)
ax.plot(x_list[:, 0], x_list[:, 1], f(x_list.T), color='black', linewidth=2)

plt.show()
