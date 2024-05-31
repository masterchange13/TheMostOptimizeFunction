#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :solve.py
# @Time      :2024/5/31 下午10:55
# @Author    ：mao

import numpy as np
import matplotlib.pyplot as plt

def draw_bagin(T, G):
    # plt.plot(T, G, 'r-')
    plt.scatter(T, G)
    plt.savefig('./graph/scatter.png')

def modify_graph(T, G):
    z1 = np.polyfit(T, G, 2)
    p1 = np.poly1d(z1)
    fx = p1(T)

    plt.plot(T, fx)
    plt.savefig('./graph/linear.png')

if __name__ == '__main__':
    T = np.array([20, 500, 1000, 1200, 1400, 1500])
    G = np.array([203, 197, 191, 188, 186, 184])
    draw_bagin(T, G)
    modify_graph(T, G)