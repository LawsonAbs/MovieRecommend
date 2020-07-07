import tools.util as ut
from algorithms import similarity as sim
from business import recommend as rec
from business.movie import Item

# 测试userCF 算法
def testUserCF():
    userList = []  # 用于存放所有的User
    ut.readFromCsvToUserData("/Users/gamidev/program/resources/ml-25m/example.csv", userList)
    # ut.printUserRate(userList)
    # 得到用户的相似度 => 注意这里的循环遍历方式，因为想避开重复的相似度计算，所以不是朴素的双层for循环
    for i in range(len(userList)):
        for j in range(i, len(userList)):
            u1 = userList[i]
            u2 = userList[j]
            sim.cosSimiForUser(u1, u2)

    ut.printInfo(userList)
    # 根据inputId 得到用户
    rec.userCFRecom(userList[0])

# 测试itemCF 算法
def testItemCF():
    itemDict = {} # 用于创建一个空字典。每个电影id为键，[(userid,rate)]为值
    itemList = [] # 用户放item 的实际对象
    userList = [] # 存放获取的所有user，存放的是User 类型
    filePath = "/Users/gamidev/program/resources/ml-25m/exam.csv"
    ut.readFromCsvToUserData(filePath,userList) #即使在itemCF的算法中，也必须要读取各个user的信息，所以需要调用这个方法
    ut.readFromCsvToItemData(filePath, itemDict)

    # 打印itemDict，类似：(98809, [(43926, 4.0), (43942, 3.0), (43955, 5.0), (43971, 4.5), (43975, 4.0)])
    for item in itemDict.items():
        movie = Item(item[0]) # 获取键，并为其创建一个实例
        for rate in item[1]: # 遍历其中的每个值
            movie.rateInfo.setdefault(rate[0],rate[1])
        itemList.append(movie) # 将movie对象 放入到itemList 中，供以后使用
        # print(item)

    # 计算每个item(在此就是movie)的相似度 => 注意这里的循环遍历方式，因为想避开重复的相似度计算，所以不是朴素的双层for循环
    for i in range(len(itemList)):
        for j in range(i, len(itemList)):
            i1 = itemList[i]
            i2 = itemList[j]
            sim.cosSimiForItem(i1, i2) # 计算出了item 间的相似度

    # 开始为用户推荐商品 => 计算与该用户正反馈列表相关的商品
    # 这里视有过打分的商品即是正反馈相关商品
    user = userList[1] # 计算该用户的推荐产品
    print(user.rateInfo) # 该用户看过的电影

    # 得到每部电影的相似电影信息，放入到一个字典中
    rate = {} # 键：movieId ，值：相似度
    for movie in itemList:
        rate.setdefault(movie.itemId,movie.simItem) # 放入一个list

    # user.rateInfo为正反馈列表
    for it in user.rateInfo.items(): # 是一个键值对
        movieId = it[0] # 拿到该用户的对每一个电影（是一个对象）的评分信息
        # 做累加得到关于每个电影的推荐值
        # rate[moviedId]是一个字典。 rate[movieId].items()则是得到许多个tuple，每个tuple里装的是与该电影的相似电影信息
        for info in rate[movieId].items():
            a,b = info[0:2]
            if a in user.movies_read: # 排除掉已经看过的电影
                continue
            user.movies_calc.setdefault(a,0) # 设置为0
            val = user.movies_calc.get(a) # 获取其值
            user.movies_calc[a] = val+b

    # 进行排序
    res = sorted(user.movies_calc.items(), key=lambda d: d[1], reverse=True)
    print("为用户%s推荐的电影有：" %user)
    print(res[0:6]) # 输出前5个推荐结果

if __name__ == '__main__':
    testItemCF()