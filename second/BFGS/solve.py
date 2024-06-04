#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :solve.py
# @Time      :2024/6/4 上午10:37
# @Author    ：mao

import numpy as np
import matplotlib.pyplot as plt
import sympy

# init value
def init_function():
    lr = 1e-3
    espi = 1e-5
    x0 = sympy.Matrix([[1],[1]])
    x = sympy.Matrix([[100],[100]])

    return lr, espi, x0, x

# function to use
def function():
    x, y = sympy.symbols('x y')
    # func = sympy.log( 1 + (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2 ) / 10
    func = sympy.log(1 + (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2) / 10
    return func


# get direction of loss
def get_dk(f):
    symbols = f.free_symbols
    count = len(symbols)

    matrix = sympy.zeros(count, 1)

    for i, symbol in enumerate(symbols):
        matrix[i, 0] = f.diff(symbol)

    return -matrix


if __name__ == '__main__':

    x, y = sympy.symbols('x y')
    lr, espi, x0, x1 = init_function()
    f = function()

    y0 = f.subs({x:x0[0], y:x0[1]})
    y = f.subs({x:x1[0], y:x1[1]})

    while (y - y0 > espi):
        dk = get_dk(f)

        x1 = x0 + lr*dk
        print(x1)

        y0 = y
        y = f(x1)

    print("x is {}", x1)
    print("y is {}", y)