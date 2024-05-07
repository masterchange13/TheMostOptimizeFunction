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
    return nn.MSELoss()(y, y_pred)

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
        x[i] = float(tmp_value) + h
        fxh1 = f(x)

        x[i] = tmp_value - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp_value

    return grad