"""
@Created by Mao on 2024/5/19
@Author:mao
@Github:https://github.com/masterchange13?tab=projects
@Gitee:https://gitee.com/master_change13
 
@File: graphOfFunction.py
@IDE: PyCharm
 
@Time: 2024/5/19 16:34
@Motto:不积跬步无以至千里，不积小流无以成江海，程序人生的精彩需要坚持不懈地积累！
@target: 大厂offer，高年薪

@@ written by GuangZhi Mao

@from:
@code target:
"""

import function as func
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = [[(i, i) for i in range(100)] for j in range(100)]
    x = np.array(x)
    # print(x)
    y = func.f(x)

    plt.plot(x[:][0], x[:][1])
    plt.show()
[0], x[:][1])
    plt.show()
