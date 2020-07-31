"""
1.使用plot工具在同一页面下进行连续画图
01.画直线的思路是：描点
"""
import matplotlib.pyplot as plt
import random
import torch
import numpy as np

num = 0  # 计数
#plt.ion()  # 开启一个画图的窗口进入交互模式，用于实时更新数据
while num < 10:
    plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
    x = np.linspace(-3, 3, 50) # 抽数
    w = random.randint(1,8)  # 随机生成一个数，作为斜率
    y = w * x + 1
    # plt.figure(num=3, figsize=(8, 5))  # figsize的设置长和宽
    plt.plot(x, y, color='red', linewidth=2.0, linestyle='dashdot')
    plt.xlim((-1, 2))  # 设置x轴的范围
    plt.ylim((-2, 3))  # 设置y轴的范围
    plt.xlabel('x')  # 设置x轴的名称
    plt.ylabel('y')  # 设置y轴额名称
    new_ticks = np.linspace(-1, 2, 5)
    plt.xticks(new_ticks)  # 设置x轴的范围的刻度值
    # 设置y轴的范围的刻度值
    plt.yticks([-2, -1, 0, 1, 2, 3]),
               #[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
               # 如果去掉上面这行的注释，那么就会将y轴的值变成上面这个数组中的值

    plt.pause(0.4)  # 设置暂停时间，太快图表无法正常显示
    num = num + 1

#plt.ioff()  # 关闭画图的窗口，即关闭交互模式
plt.show()  # 显示图片，防止闪退
