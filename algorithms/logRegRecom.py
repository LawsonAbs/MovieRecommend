"""逻辑回归推荐
"""
import torch as t
import torch.nn as nn
from torch.utils.data import(
    Dataset,
    DataLoader
)
import datetime as dt
from tools import jsonUtil  # 导入加载数据的包

BATCH_SIZE = 5
TRAIN_EPOCH = 20
TEST_EPOCH = 10

class LogRegRecom(nn.Module):
    def __init__(self,inFeatures,outFeatures):
        """
        :param inFeatures: 代表输入数据的维度中的最后一个值。比如如果数据x的维度是[2,6]，那么这里的inFeatures就是6
        :param outFeatures: 代表输出数据的维度中的最后一个值。比如如果数据x的维度是[6,512]，那么这里的inFeatures就是512
        """
        super(LogRegRecom,self).__init__()
        self.linear = nn.Linear(inFeatures,outFeatures) # 执行线性变化操作
        self.sigmoid = nn.Sigmoid() # 执行sigmoid操作

    # x是特征输入数据
    def forward(self,x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

# 读取&存储数据
class GamiData(Dataset):
    def __init__(self,userInfo,interact,dispatchInfo):
        super().__init__()
        # 这里存储特征数据，标签数据
        self.data,self.label= jsonUtil.getWholeData(userInfo,interact,dispatchInfo)

    def __getitem__(self, index: int):
        return t.tensor(self.data[index]).type(t.FloatTensor),t.tensor(self.label[index]).type(t.FloatTensor)

    def __len__(self):
        return len(self.data)


# 开始执行训练操作
def train(gamiData,modelPath):
    train_loader = DataLoader(gamiData,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=0)
    logr = LogRegRecom(33,1)
    criterion = nn.BCELoss() # 交叉熵计算损失
    optimizer = t.optim.Adam(logr.parameters(), lr=1e-4)

    # step3.开始训练
    # 每个epoch用的都是同一批数据进行训练
    for epoch in range(TRAIN_EPOCH):
        print("=========epoch：", epoch + 1, end=",")
        right = 0  # 记正确数
        # enumerate
        for i, item in enumerate(train_loader):
            # print(type(item), "++=====") <class 'list'>
            _da, label = item
            # print("type(_da) === ",type(_da)) # <class 'torch.Tensor'>
            out = logr(_da)
            outSize= out.size(0)
            out = out.view(outSize)  # 要调整一下，才能跟后面的label进入到BCELoss()的部分
            loss = criterion(out, label)  # 与分类标签做比较，求出损失

            # ge()函数的使用，可以参考pytorch doc文档
            mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
            correct = (mask == label).sum()  # 计算正确预测的样本个数
            right += correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total = len(train_loader) * BATCH_SIZE  # 记总数
        print("acc = ", right / total)

    curTime = dt.datetime.now()
    curTime = curTime.strftime("%Y%d%m_%H%M%S")
    modelPath = modelPath + curTime + ".abc"
    t.save(logr.state_dict(), modelPath)  # 保存最后的训练模型


def recom(modelPath,input_loader):
    """
    功能：使用训练好的模型，把值得推荐的dispatch放到 dispatchId List中
    :param modelPath: 加载训练好的模型
    :param input: 待输入的数据
    :return:dispatchIdList:返回推荐的 dispatchId list。
    """
    predict = LogRegRecom(33,1)
    predict.load_state_dict(modelPath)  # 从指定的modelPath 中获取训练好的模型
    dispatchIdList = []
    for i, item in enumerate(input_loader):
        _da = item
        out = predict(_da)
        mask = out.ge(0.5).float()
        if mask == 1:
            dispatchIdList.append(_da[0]) # 获取dispatchId
    return dispatchIdList


def doTrain():
    # part1.训练模型
    # step1.初始化路径信息
    userFilePath = "/Users/gamidev/program/recom_gami/data/gami/user_attr.json"  # 用户数据
    trackFilePath = '/Users/gamidev/program/recom_gami/data/gami/track.json'  # 交互数据
    dispatchFilePath = "/Users/gamidev/program/recom_gami/data/gami/dispatch.json"  # 派发信息
    productFilePath = "/Users/gamidev/program/recom_gami/data/gami/product.json"  # 商品信息
    modelPath = ""  # 保存训练好的模型

    # step2.获取各种数据
    userInfo = jsonUtil.readUserJson2Dict(userFilePath)
    interact = jsonUtil.readTrackJson2Dict(trackFilePath)
    prodcutInfo = jsonUtil.readProductJson2Dict(productFilePath)
    dispatchInfo = jsonUtil.readDispatch2Dict(dispatchFilePath, prodcutInfo)
    gamiData = GamiData(userInfo, interact, dispatchInfo)

    # step3.训练模型
    train(gamiData, modelPath)


def doTest():
    """
    这部分的任务是用于做测试
    :return:
    """
    # part2.进行预测
    # step1.准备数据
    # step2.预测
    pass


if __name__ == "__main__":
    doTrain()