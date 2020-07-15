import csv
import numpy as np

from business.user import User
from business.movie import Item

moviesRead =[] #已看过的电影集合
rating = {}  # 初始化一个dict，存储的是 {movie:rate}
curUserId = 1
user = User(1, '1', '1')  # 初始化的信息，但需要注意这里的
curMovieId = 1 # 当前的movieId
movie = Item(1) # 传入参数进行构造
dic = {}
userList = []
"""
功能：读取csv文件
Parms:filePathName: 需要读取的文件地址（注意只有输入 "/Users/gamidev/program/MovieRecommend/resources/ml-25m/ratings.csv" 这样
的内容才算正确）
cfStyle: 表明
"""
def readFromCsvToUserData(filePathName,userList):
    # 判断文件路径是否正确
    with open(filePathName) as file:  # 这句话是什么意思 => 打开filePathName所指的那个文件，然后将其存储在文件对象file中
        ratingReader = csv.reader(file)
        next(ratingReader)  # 摆脱第一行的数据
        for line in ratingReader:
            # 为每个用户形成一个字典
            userExtract(line, userList)
    user.rateInfo = rating
    userList.append(user)

"""
从每行中提取有效信息，形成一个字典;
这个line应该是一个列表
为用户形成数据信息，故为userExtract()
"""
def userExtract(line, userList):
    userId, movieId, rate = line[0:-1]
    userId = int(userId)
    movieId = int(movieId)
    global curUserId, user, rating, moviesRead
    if userId != curUserId:
        # step 1.赋值
        user.moviesRead = moviesRead
        user.rateInfo = rating
        if rating: # 防止加入空用户
            userList.append(user)  # 清空dict 中的值

        # step 2.重置
        user = User(userId, str(userId), str(userId))  # 新建一个User
        curUserId = userId
        rating = {}
        moviesRead = [] #
    moviesRead.append(movieId)
    rating[int(movieId)] = float(rate)  # 转换为数字


"""
01.读取数据形成一个MovieData
"""
def readFromCsvToItemData(filePathName,itemDict):
    # 判断文件路径是否正确
    with open(filePathName) as file:  # 这句话是什么意思 => 打开filePathName所指的那个文件，然后将其存储在文件对象file中
        ratingReader = csv.reader(file)
        next(ratingReader)  # 摆脱第一行的数据
        for line in ratingReader:
            itemDict = itemExtract(line,itemDict)

"""
01.基于 itemCF 算法进行抽取，形成的是一个字典，字典形式如下：
"""
def itemExtract(line,itemDict):
    userId, movieId, rate = line[0:-1]
    userId = int(userId)
    movieId = int(movieId)
    rate = float(rate)
    itemDict.setdefault(movieId,[]).append((userId,rate)) # 往字典中添加tuple
    return itemDict


# 打印用户的评分情况
def printUserRate(userList):
    for user in userList:
        print("用户", user, "的评分情况如下：")
        for key, val in user.rating.items():  # 遍历
            print(key, "->", val, end="| ")
        print()


# 根据向量得到模长【其实传入的是一个字典】
def getLenOfVector(dict):
    sum = 0
    for value in dict.values():
        sum += (value * value)
    sum = np.sqrt(sum)
    return sum

# 打印可供推荐的用户Id
def printInfo(userList):
    print("你可以使用的Userid 有：")
    start = 10
    end = 15
    for i in userList[start:end]:
        print(i,end =" ")
    print()

# 得到一个共现矩阵 => 理应不需要使用这个共现矩阵，因为这会导致矩阵过于庞大，还是使用有数便存的方式
def getCooccMatrix():
    pass