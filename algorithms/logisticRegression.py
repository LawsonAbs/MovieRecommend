import torch as t
from torch import nn


class Linear(nn.Module):  # 继承nn.Module
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()  # 等价于调用 nn.Module.__init__(self)
        self.w = nn.Parameter(t.randn(in_features, out_features))  # 以(in_features,out_features)为形状初始化一个参数w
        self.b = nn.Parameter(t.randn(out_features))  # 以 out_features 这个形状初始化一个参数b

    def forward(self, x):
        x = x.mm(self.w)  # x.@(self.w)
        return x + self.b.expand_as(x)

layer = Linear(4, 3)
print(layer.type)
print(layer)
input = t.randn(2, 4)
output = layer(input)  # 还是那个问题，这里的layer本是个对象，但是却直接被用作函数了，