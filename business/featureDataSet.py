import os
from torch.utils.data import Dataset,DataLoader
import  pandas as pd

BATCH_SIZE = 5 # 批大小

"""
特征数据集
1.用于生成"用于训练的特征向量"
"""
class FeatureDataset(Dataset):
    def __init__(self,userRateFilePath,userInfo): # userInfo 是一个事先处理好的字典
        if not os.path.isfile(userRateFilePath):
            return ValueError(userRateFilePath,"is'n a file")
        self.filePath = userRateFilePath
        self.feature = [] # 评分信息+用户信息+其它 得到的一个特征向量
        self.userInfo = userInfo

    # 根据文件路径获取文件内容
    def loadData(self):
        with open(self.filePath) as file:
            for line in file.readlines():
                line = line.strip("\n")
                print(line)
        return

    def __len__(self):
        return len(self.feature)

    def __getitem__(self,idx):
        return self.feature[idx]  # 返回这个特征

# 直接传入一个类或者是传入一个类的实例
data = DataLoader(FeatureDataset,
                  batch_size = BATCH_SIZE,
                  shuffle=True)
