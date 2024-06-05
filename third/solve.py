"""
@Created by Mao on 2024/6/5
@Author:mao
@Github:https://github.com/masterchange13?tab=projects
@Gitee:https://gitee.com/master_change13
 
@File: solve.py
@IDE: PyCharm
 
@Time: 2024/6/5 13:36
@Motto:不积跬步无以至千里，不积小流无以成江海，程序人生的精彩需要坚持不懈地积累！
@target: 大厂offer，高年薪

@@ written by GuangZhi Mao

@from:
@code target:
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义符号变量
x1, x2, x3 = sp.symbols('x1 x2 x3')

# 定义目标函数
f = -2 * x1 + x2 - 3 * x3

# 定义约束条件
constraints = [
    x1 + x2 + x3 - 1,
    -x1,
    -x2,
    -x3
]

# 将目标函数和约束条件转换为可计算的函数
f_func = sp.lambdify((x1, x2, x3), f, 'numpy')
constraints_funcs = [sp.lambdify((x1, x2, x3), constraint, 'numpy') for constraint in constraints]

# 目标函数带障碍项
def objective_with_barrier(x, mu):
    penalty = sum([-mu * np.log(-constraint(x[0], x[1], x[2])) for constraint in constraints_funcs])
    return f_func(x[0], x[1], x[2]) + penalty

# 梯度下降法
def gradient_descent_with_barrier(x0, mu, alpha, tol, max_iter):
    path = [x0]
    x = np.array(x0, dtype=float)

    for i in range(max_iter):
        grad = np.array([sp.diff(f + sum([-mu * sp.log(-constraint) for constraint in constraints]), var).subs({x1: x[0], x2: x[1], x3: x[2]}).evalf() for var in [x1, x2, x3]], dtype=float)
        x_new = x - alpha * grad
        path.append(x_new)

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, path

# 初始值和参数设置
x0 = [0.3, 0.3, 0.4]
mu = 1.0
alpha = 0.01
tol = 1e-6
max_iter = 100

# 优化
solution, path = gradient_descent_with_barrier(x0, mu, alpha, tol, max_iter)

# 输出结果
print("最小值点：", solution)
print("最小值：", f_func(solution[0], solution[1], solution[2]))

# 可视化优化路径
path = np.array(path)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制优化路径
ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r.-', markersize=10, label='Optimization Path')

x1_vals = np.linspace(0, 1, 100)
x2_vals = np.linspace(0, 1, 100)
x3_vals = np.linspace(0, 1, 100)
X1, X2, X3 = np.meshgrid(x1_vals, x2_vals, x3_vals)
Z = f_func(X1, X2, X3)
ax.scatter(X1, X2, X3, c=Z, cmap='viridis', alpha=0.1)

ax.set_xlabel('x1', fontsize=14)
ax.set_ylabel('x2', fontsize=14)
ax.set_zlabel('x3', fontsize=14)
plt.title('Optimization Path Visualization', fontsize=16)
plt.legend()
plt.savefig("./graph/result.png")
plt.show()