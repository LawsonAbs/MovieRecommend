"""
1.先针对单一用户做一个逻辑回归
01.读取数据 => u.data_exam.csv ，形成向量
向量的形式如下：
[userId,age,gender,occ,movieId,movieTopic,]
02.开始逻辑回归的操作
03.进行一波预测
"""
import sys
sys.path.append(r'.') # 将当前环境添加到系统环境中，用于给Python找包
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import datetime as dt  # 用于得到时间
import torch as t
from torch import nn

from torch.utils.data import Dataset,DataLoader
from tools import utils as ut
import tools.printUtil  as pu

"""
1.数据文件 u.data_exam 中有40行数据，其数据格式如下：
 ...
 32	301	98	4	882075827
 ...
"""
BATCH_SIZE = 5  # 批大小【每批的数据个数】
EPOCH_TRAIN = 10
EPOCH_TEST = 10

# 推荐使用绝对路径，否则会在ide和命令行间产生错误
rateFilePath = "/Users/gamidev/program/MovieRecommend/data/ml-100k/u1.base"  # 用户评分数据
userInfoPath = "/Users/gamidev/program/MovieRecommend/data/ml-100k/u.user"  # 用户信息数据
movieInfoPath = "/Users/gamidev/program/MovieRecommend/data/ml-100k/u.item"  # 电影信息数据
occupation2Id = {}  # the mapping occupation  to id


#初始化常量设置
userInfo = ut.getUserInfo(userInfoPath)  # 得到用户的基本信息 => 事先处理好，形成一个字典
movieInfo = ut.getMovieInfo(movieInfoPath)  # 得到电影的基本信息

"""
特征数据集
1.用于生成"用于训练的特征向量"
"""
class FeatureDataset(Dataset):
    def __init__(self,userRateFilePath,userInfo):
        if not os.path.isfile(userRateFilePath):
            return ValueError(userRateFilePath,"isn't a file")
        self.filePath = userRateFilePath

        # 评分信息+用户信息+其它 得到的一个特征向量  =》 这里面报存的就是每条训练数据
        # feature 中的每条都是一个list
        self.feature = [] # 特征向量数据
        self.userInfo = userInfo
        self.label = [] # 标签数据，初始化为0，表示不感兴趣

    # 根据文件路径获取文件内容
    def loadData(self):
        with open(self.filePath) as file:
            for line in file.readlines():  # 得到的是评分信息的每一行
                line = line.strip("\n")
                rateInfo = []
                for row in line.split()[0:-1]:
                    row = row.strip() # 去前后空格
                    rateInfo.append(int(row))
                # movieInfo   # 得到电影的数据
                singleUser = self.userInfo.get(str(rateInfo[0]))  # 得到电影的id 所对应的单个用户的特征信息
                age = int(singleUser['age']) # 年龄
                gender = singleUser['gender'] # 性别
                occ = singleUser['occ'] # 职业
                #print (singleUser)

                # ==== 调整用户信息数据 ====
                age = int(age)
                gender = 0 if gender=='M' else 1 # 如果是男性，则是0；否则为1
                if occ in occupation2Id.keys(): # 如果在其中
                    occ = occupation2Id.get(occ) #
                else: # 否则，重新生成一个值
                    occupation2Id[occ] = len(occupation2Id)  # 构建一个occupation -> id 的字典
                    occ = len(occupation2Id) - 1 # 一个新的职业
                data = [rateInfo[0],age,gender,occ] # 拼接用户的数据成一个list

                # ==== 调整电影信息数据 ====
                singleMovie = movieInfo.get(rateInfo[1]) # 得到指定电影id的电影信息
                movieId = singleMovie['id']
                movieTopic = singleMovie['topic']

                # ============下面是拼接所有的数据成为一个feature============
                data.append(int(movieId))  # 追加了一个电影Id
                for a in movieTopic:
                    data.append(a)  # 追加了一个topic序列
                if rateInfo[2] >= 2:  # 如果是评分大于等于2，则标签为1 => 发现这个对训练的效果影响很大
                    self.label.append(1)  # 修改为1
                else:
                    self.label.append(0)  # 返回为0
                self.feature.append(data)  # 追加到feature 中

    def __len__(self):
        return len(self.feature)

    def __getitem__(self,idx):
        # 返回某条特征向量 + 标签向量
        # return self.feature[idx], self.label[idx]  => 必须得返回tensor，否则是不可以的
        # 但是貌似有个问题：这里的self.feature 好像在运行之前就把数据加载了，有没有一种方法是在运行时返回数据？
        return t.tensor(self.feature[idx]).type(t.FloatTensor), t.tensor(self.label[idx]).type(t.FloatTensor)


# 逻辑回归类
class LogR(nn.Module):  # 继承nn.Module
    def __init__(self, in_features, out_features):
        super(LogR, self).__init__()  # 等价于调用 nn.Module.__init__(self)
        # 使用线性回归
        # 这里的24 的含义是：向量的维度是BATCH_SIZE * 24  => 最后只输出一个数，所以输出维度为1
        # 套路就是：在__init__()方法中定义变量，获取类实例等等，然后再在forward()中调用
        self.linR = nn.Linear(in_features,out_features)
        self.sg = nn.Sigmoid()  # 输出之后执行sigmoid()函数
        
    def forward(self, x): 
        # 分别执行操作
        x = self.linR(x)
        x = self.sg(x)
        return x

def train(modelPath):
    featureDs = FeatureDataset(rateFilePath,userInfo)  # 获取特征向量
    featureDs.loadData()  # 手动加载数据
    train_loader = DataLoader(featureDs,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=0)  # 如果值为0，则表示只用主进程加载数据

    # 判断是否可以用GPU 加速 ============
    # 注意这里  cuda:0 指的
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    for k,v in train_loader:
        if type(v) != list:   # 数据转到GPU上
            v.to(device)

    # ============ 开始训练 ============
    logr = LogR(24, 1)  # 特征向量是24*1维
    logr.to(device) # 模型放到cuda 上

    # 定义损失函数 + 优化器
    criterion = nn.BCELoss()  # 交叉熵函数作为计算损失
    criterion.to(device)
    # optimizer = t.optim.SGD(logr.parameters(), lr=1e-3, momentum=0.9) # 在本代码中使用SGD训练，效果不好
    optimizer = t.optim.Adam(logr.parameters(), lr=1e-4)

    # step3.开始训练
    # 每个epoch用的都是同一批数据进行训练
    for epoch in range(20):
        print("=========epoch：", epoch + 1, end = ",")
        right = 0  # 记正确数
        # enumerate
        for i, item in enumerate(train_loader):
            # print(type(item), "++=====") <class 'list'>
            _da, label = item
            # print("type(_da) === ",type(_da)) # <class 'torch.Tensor'>
            _da.to(device) # 将数据放到指定的 device
            label.to(device)
            out = logr(_da)
            """
            01.需要注意，这里并不是对BATCH_SIZE进行view，因为有的时候数据并不一定能整除BATCH_SIZE
            但是能整除out.size(0)是肯定的
            02.要调整一下，才能跟后面的label进入到BCELoss()的部分            
            """
            outSize = out.size(0)
            out = out.view(outSize)
            out = out.to(device)  # 要将out 放到device  中，否则最后会有一个报错  => 在学院的gpu上，就发现这个才是最重要的！
            loss = criterion(out, label)  # 与分类标签做比较，求出损失
            # type(loss) =  <class 'torch.Tensor'> 这句代码的作用是将单个值的tensor 转为一个python中的数值
            print_loss = loss.data.item()
            mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
            correct = (mask == label).sum()  # 计算正确预测的样本个数
            right += correct.item()
            # acc = correct.item() / _da.size(0)  # 计算精度
            # print("loss = ",loss,"acc=",acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total = len(train_loader) * BATCH_SIZE  # 记总数
        print("acc = ", right / total)

    curTime = dt.datetime.now()
    curTime = curTime.strftime("%Y%m%d_%H%M%S")
    modelPath = modelPath + curTime + ".abc"
    t.save(logr.state_dict(), modelPath)  # 保存最后的训练模型


# 模型测试部分
def test(testRateFilePath,modelPath):
    feature = FeatureDataset(testRateFilePath, userInfo)  # 获取特征向量
    feature.loadData()  # 手动加载数据
    test_loader = DataLoader(feature,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=2)  # 如果值为0，则表示只用主进程加载数据

    # ======= 判断是否可以用GPU 加速 ============
    # 注意这里  cuda:0 指的
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    for k, v in test_loader:
        if type(v) != list:  # 数据转到GPU上
            v.to(device)

    # ============ 开始训练 ============
    logr = LogR(24, 1)  # 特征向量是24*1维
    logr.to(device)  # 模型放到cuda 上
    criterion = nn.BCELoss()
    criterion.to(device)
    logr.load_state_dict(t.load(modelPath))  # 保存最后的训练模型

    right = 0  # 记正确数
    for i, item in enumerate(test_loader):
        _da, label = item
        _da.to(device)  # 将数据放到指定的 device
        label.to(device)
        out = logr(_da)
        out = out.view(BATCH_SIZE)  # 要调整一下，才能跟后面的label进入到BCELoss()的部分
        out = out.to(device)  # 要将out 放到device  中，否则最后会有一个报错  => 在学院的gpu上，就发现这个才是最重要的！
        loss = criterion(out, label)  # 与分类标签做比较，求出损失
        print_loss = loss.data.item()
        mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
        correct = (mask == label).sum()  # 计算正确预测的样本个数
        right += correct.item()

    total = len(test_loader) * BATCH_SIZE  # 记总数
    print("acc = ", right / total)



    print("=================================")
if __name__ == "__main__":
    sys.argv.extend([BATCH_SIZE,EPOCH_TRAIN,EPOCH_TEST])
    pu.printConfig(sys.argv)
    if len(sys.argv) <= 1:
        print("参数不足")
        exit(0)
    # =========== 在训练集上测试 ===========
    elif sys.argv[1] == "test":
        rateFilePath = sys.argv[2] # 拿到测试数据集
        modelPath = "/Users/gamidev/program/MovieRecommend/checkpoint/20200508_104227.abc"
        test(rateFilePath,modelPath)
    # =========== 在测试集上进行训练 ============
    else:
        rateFilePath = sys.argv[2] # 拿到训练数据
        modelPath = "/Users/gamidev/program/MovieRecommend/checkpoint/"
        train(modelPath)