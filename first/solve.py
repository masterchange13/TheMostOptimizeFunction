# -*- coding:utf-8 _*-
"""
@Created by Mao on 2024/5/7 14:35
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
import random

import torch
import torch.nn as nn
import cv2
import numpy as np
import function as func
import matplotlib.pyplot as plt

def make_data_to_graph(point, point_y, loss_g):
    plt.subplot(1, 2, 1)
    plt.plot(point, point_y, 'ro-')
    plt.subplot(1, 2, 2)
    plt.plot(point, loss_g, 'b-')
    plt.show()
    return 0

if __name__ == '__main__':

    """"
        init value
    """
    lr = 1e-4
    thredhold = 1e-4
    x = np.array([100, 100])
    y_pred = np.array([1, 1])

    f_x, x, point, point_y, loss_g, real_count = func.solve_threshold(func.f, lr, thredhold, x, y_pred)
    print("f_x is {}", f_x)
    print("x is {}", x)

    print(point)
    print(point_y)
    print("real_count is {}", real_count)

    make_data_to_graph(point, point_y, loss_g)

    make_data_to_graph(point, point_y, loss_g)