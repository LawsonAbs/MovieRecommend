"""

1.训练数据：
训练数据包括了y轴的数据，即是一个点集 (x,y)。我之前一直都以为是用x做训练数据，这是大错特错的。
不能单纯的以为写的是 y = Xw+b，就以为X只是x轴对应的值，这是偏见。
同时，将y轴作为一个上式 X 的一部分的原因是：y轴的值同x轴值一样也是一个特征。所以特征为 X = (x,y)
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# step 0.准备数据 =》 模拟生成
n_data = torch.ones(500, 2)  # 数据的基本形态
data0 = torch.normal(2 * n_data, 1)
data1 = torch.normal(-2 * n_data, 1)
class0 = torch.zeros(500)
class1 = torch.ones(500)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
data = torch.cat((data0, data1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
cla = torch.cat((class0, class1), 0).type(torch.FloatTensor)  # LongTensor = 64-bit integer


# step 1.定义自己的模型（当然是要从nn.Module这个包中继承而来）
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)  # 定义线性变换
        self.sm = nn.Sigmoid()  # 定义sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


# step2.得到模型实例；定义损失函数和优化器
logistic_model = LogisticRegression()
criterion = nn.BCELoss()
# 对logistic_model 这个模型的中的参数进行优化，同时赋予学习率和momentum
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)

# step3.开始训练
# 每个epoch用的都是同一批数据进行训练
for epoch in range(500):
    # train data =>训练数据
    _da = data
    cla_data = cla  # 分类标签信息

    out = logistic_model(_da)
    loss = criterion(out, cla_data)  # 与分类标签做比较，求出损失
    # for name,para in logistic_model.named_parameters():
    #     print(name,para)
    # print(_da)
    # print(out)  # 这里计算对不上的原因是，后面还有一个sigmoid()函数对xw+b进行处理才能得到out

    # type(loss) =  <class 'torch.Tensor'> 这句代码的作用是将单个值的tensor 转为一个python中的数值
    print_loss = loss.data.item()
    mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
    correct = (mask == cla_data).sum()  # 计算正确预测的样本个数
    acc = correct.item() / data.size(0)  # 计算精度
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每隔20轮打印一下当前的误差和精度  => 并打印
    if (epoch + 1) % 20 == 0:
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        print('acc is {:.4f}'.format(acc))  # 精度

        # step 4.结果可视化
        plt.clf()  # 清空
        w0, w1 = logistic_model.lr.weight[0]
        w0 = float(w0.item())
        w1 = float(w1.item())
        b = float(logistic_model.lr.bias.item())
        plot_x = np.arange(-7, 7, 0.1)
        plot_y = (-w0 * plot_x - b) / w1
        plt.scatter(_da.data.numpy()[:, 0], _da.data.numpy()[:, 1], c=cla_data.data.numpy(), s=100, lw=0, cmap='RdYlGn')
        plt.plot(plot_x, plot_y)
        plt.pause(1)  # 暂停1s

plt.show()
plt.close()
