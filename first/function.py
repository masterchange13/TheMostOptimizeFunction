# -*- coding:utf-8 _*-
"""
@Created by Mao on 2024/5/7 14:29
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
import numpy as np
import pandas as pd
from utils import *

"""
    x：矩阵
    y：矩阵
"""
def f(data):
    x = data[0]
    y = data[1]
    part1 = 1 + pow((1.5 - x + x*y), 2)
    part2 = pow((2.25 - x + x*y**2), 2)
    part3 = pow((2.625 - x + x*y**3), 2)

    return np.log(part1 + part2 + part3) / 10

"""
    指定迭代次数进行最优化求解
    f:function
    lr: learning rate
    threshold: 阈值
    x: init data
    epoch: epoch count
"""

# def sovle(f, lr, threshold, x, epoch):
#     grad
#     for i in range(epoch):


"""
    根据梯度下降法求解，通过阈值判断结束
    f: function
    lr: learning rate
    threshold: 阈值
    x: init data
    
    target: get the best x for f
"""
def solve_threshold(f, lr, threshold, x, y_pred):

    """
        画出图形
            point
            point_y
            loss_g
    """
    point = []
    point_y = []
    loss_g = []
    count = 0
    real_count = 0

    grad = gradient(x)
    x = update_value(x, lr, grad)
    new_y = f(x)
    loss = get_loss(new_y, y_pred)
    while loss > threshold:
        grad = gradient(x)
        x = update_value(x, lr, grad)
        new_y = f(x)
        loss = get_loss(new_y, y_pred)

        # update lr
        if count >= 1000:
            lr = lr / 10
            count = 0

        point.append(x)
        point_y.append(new_y)
        loss_g.append(loss)
        count += 1
        real_count += 1

        print(loss)

    return f(x), x, point, point_y, loss_g


