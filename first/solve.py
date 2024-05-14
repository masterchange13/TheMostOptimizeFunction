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
    plt.plot(point, point_y, 'ro')
    plt.plot(point, loss_g, 'b-')
    plt.show()
    return 0

if __name__ == '__main__':

    """"
        init value
    """
    lr = 1e-3
    thredhold = 1e-2
    x = np.random.random([2])
    y_pred = np.array([1, 1])

    f_count, x_count, point, point_y, loss_g = func.solve_threshold(func.f, lr, thredhold, x, y_pred)
    print(f_count)
    print(x_count)

    make_data_to_graph(point, point_y, loss_g)