"""
1.先针对单一用户做一个逻辑回归
01.读取数据 => lr_example.csv ，形成向量
02.开始逻辑回归的操作
03.进行一波预测
"""

import torch as t
from torch import nn


class Linear(nn.Module):  # 继承nn.Module
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()  # 等价于调用 nn.Module.__init__(self)

    def forward(self, x):
        pass

layer = Linear(4, 3)
print(layer.type)
print(layer)
input = t.randn(2, 4)
output = layer(input)  # 还是那个问题，这里的layer本是个对象，但是却直接被用作函数了