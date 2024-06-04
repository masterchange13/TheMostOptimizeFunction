#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test.py
# @Time      :2024/6/4 上午11:00
# @Author    ：mao

import sympy
import numpy as np
import matplotlib.pyplot as plt

def function():
    x, y = sympy.symbols('x y')
    func = sympy.log( 1 + (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2 ) / 10
    return func

def get_func_count():
    f = function()
    free_symbols = f.free_symbols
    # print("free_symbols:", free_symbols)
    return free_symbols

if __name__ == '__main__':
    symbol = get_func_count()
    print(symbol)
    print(len(symbol))
    for symbol_item in symbol:
        print(symbol_item)