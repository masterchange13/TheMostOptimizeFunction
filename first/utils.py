# -*- coding:utf-8 _*-
"""
@Created by Mao on 2024/5/7 14:58
@Author:mao
@Github:https://github.com/masterchange13?tab=projects
@Gitee:https://gitee.com/master_change13
 
@File: python_test.py
@Time: 2023
@Motto:不积跬步无以至千里，不积小流无以成江海，程序人生的精彩需要坚持不懈地积累！
@target: 大厂offer，高年薪

@@ written by GuangZhi Mao

@from:
@code target:
"""

import torch
import torch.nn as nn
import cv2
import numpy as np

"""
    工具类模块
    包括
        1. 求权重
        2. 求均值
        3. 求方差
        4. 求标准差
        5. 求损失函数
        6. 梯度更新
        7. 获取梯度
"""

"""
    1. 求权重
"""
def get_weight(f):
    return f.weight

"""
    2. 求均值
"""
def get_mean(x):
    return np.mean(x)

"""
    3. 求方差
"""
def get_variance(x):
    return np.var(x)

"""
    4. 求标准差
"""
def get_std(x):
    return np.std(x)

"""
    5. 求损失函数
"""
def get_loss(y, y_pred):
    return np.sum((y -  y_pred)**2)

"""
    6. 梯度更新
"""
def update_weight(w, lr, grad):
    w = w - lr * grad
    return w

"""
    7. 获取梯度
"""
def get_grad(f, x):
    h = 1e-5
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_value = x[i]
        print(x.dtype)
        print(tmp_value.dtype)
        x[i] = float(tmp_value) + h
        fxh1 = f(x)

        x[i] = tmp_value - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp_value

    return grad

"""
    8. update value
"""
def update_value(x, lr, grad):
    x = x - lr*grad
    return x

"""
    9 梯度公式
"""
def gradient(data):
    x, y = data
    # 计算各部分值
    part1 = 1 + np.power((1.5 - x + x * y), 2)
    part2 = np.power((2.25 - x + x * y ** 2), 2)
    part3 = np.power((2.625 - x + x * y ** 3), 2)

    # 总和用于分母
    sum_parts = part1 + part2 + part3

    # 计算关于 x 的偏导数项
    df_dx_part1 = -2 * (1.5 - x + x * y) - y * (1.5 - x + x * y) ** 2
    df_dx_part2 = -2 * (2.25 - x + x * y ** 2) - y ** 2 * (2.25 - x + x * y ** 2) ** 2
    df_dx_part3 = -2 * (2.625 - x + x * y ** 3) - y ** 3 * (2.625 - x + x * y ** 3) ** 2
    df_dx = (df_dx_part1 + df_dx_part2 + df_dx_part3) / (10 * sum_parts)

    # 计算关于 y 的偏导数项
    df_dy_part1 = 2 * x * (1.5 - x + x * y)
    df_dy_part2 = 2 * x * y * (2.25 - x + x * y ** 2)
    df_dy_part3 = 2 * x * y ** 2 * (2.625 - x + x * y ** 3)
    df_dy = (df_dy_part1 + df_dy_part2 + df_dy_part3) / (10 * sum_parts)

    return np.array([df_dx, df_dy])
 x * y ** 3)
    df_dy = (df_dy_part1 + df_dy_part2 + df_dy_part3) / (10 * sum_parts)

    return np.array([df_dx, df_dy])
