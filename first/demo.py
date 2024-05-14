# -*- coding:utf-8 _*-
"""
@Created by Mao on 2024/5/14 14:13
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
import numpy as np

# 数据集
x = np.array([0.18, 0.1, 0.16, 0.08, 0.09, 0.11, 0.12, 0.17, 0.15, 0.14, 0.13])
y = np.array([0.18, 0.1, 0.16, 0.08, 0.09, 0.11, 0.12, 0.17, 0.15, 0.14, 0.13])

# lr为学习率
lr = 0.01
# w为权值
w = 10
# epoches为循环进行的次数
epoches = 500
# grad为维度
grad = 0

list_w = []
list_loss = []
list_grad = []
list_i = []


# 进行梯度的更新
def grad_new(x, y, w):
    return np.mean((x * w - y) * w)


# 进行损失值的更新
def loss_new(x, y, w):
    return 0.5 * np.sum((w * x - y) ** 2)


for i in range(epoches):
    grad = grad_new(x, y, w)
    w = w - lr * grad
    loss = loss_new(x, y, w)
    print(f"第{i + 1}次迭代的梯度为{grad}，权值为{w}，损失值为{loss}")

    list_w.append(w)
    list_i.append(i)
    list_loss.append(loss)
    list_grad.append(grad)

# 导包
from pyecharts.charts import Line
from pyecharts.options import TitleOpts, ToolboxOpts

line1 = Line()
line1.add_xaxis(list_i)
line1.add_yaxis("梯度", list_grad)
line1.set_global_opts(
    title_opts=TitleOpts(title="梯度与迭代次数的关系", pos_left="center", pos_bottom="1%"),
    toolbox_opts=ToolboxOpts(is_show=True),
)
line1.render("梯度与迭代次数的关系.html")

line2 = Line()
line2.add_xaxis(list_w)
line2.add_yaxis("损失值", list_loss)
line2.set_global_opts(
    title_opts=TitleOpts(title="损失值与参数的关系", pos_left="center", pos_bottom="1%"),
    toolbox_opts=ToolboxOpts(is_show=True),
)
line2.render("损失值与参数的关系.html")