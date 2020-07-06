import csv
import numpy as np

from business.user import User

movies_read =[] #已看过的电影集合
rating = {}  # 初始化一个dict
curId = 1
user = User(1, '1', '1')  # 初始化的信息，但需要注意这里的
"""
功能：读取csv文件
Parms:filePathName: 需要读取的文件地址（注意只有输入 "/Users/gamidev/program/MovieRecommend/resources/ml-25m/ratings.csv" 这样
的内容才算正确）
"""
def readDataFromCsv(filePathName, userList):
    # 判断文件路径是否正确
    with open(filePathName) as file:  # 这句话是什么意思 => 打开filePathName所指的那个文件，然后将其存储在文件对象file中
        ratingReader = csv.reader(file)
        next(ratingReader)  # 摆脱第一行的数据
        for line in ratingReader:
            # 为每个用户形成一个字典
            extract(line, userList)
    # 加入最后一个user，否则会漏掉
    user.rating = rating
    userList.append(user)

# 从每行中提取有效信息，形成一个字典;
# 这个line应该是一个列表
def extract(line, userList):
    userId, movieId, rate = line[0:-1]
    userId = int(userId)
    movieId = int(movieId)
    global curId, user, rating, movies_read
    if userId != curId:
        # step 1.赋值
        user.movies_read = movies_read
        user.rating = rating
        if rating: # 防止加入空用户
            userList.append(user)  # 清空dict 中的值

        # step 2.重置
        user = User(userId, str(userId), str(userId))  # 新建一个User
        curId = userId
        rating = {}
        movies_read = [] #
    movies_read.append(movieId)
    rating[int(movieId)] = float(rate)  # 转换为数字


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