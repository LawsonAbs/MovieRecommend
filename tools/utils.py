import csv
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
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


# part2.======== 下面的代码是为 LogRegression 读取文件并形成一个向量返回的 =====
"""
params
    line:表示输入的行内容
"""
def extractUserInfo(line):
    temp = {}
    line = line.split("|")
    #print(line)
    temp['id'],temp['age'],temp['gender'],temp['occ'] = line[0:-1] # 赋值id,age,gender,occ
    return temp


"""
fun：
    获取评分信息
params
    line:表示输入的行内容
returns:
    返回结果是一个
"""
def extractRateInfo(line):
    line = line.split()
    return line


"""
1.获取单个用户的特征数据
para:读取的文件内容，从该些文件中(path)，获取用户的评分信息 + 电影信息 + 其它信息
特征向量的格式是：x=【用户id，用户年龄，性别，职业，电影id，电影类别】
针对数据：u.user => 19|40|M|librarian|02138
数据说的是：userid=19，年龄是40，性别是Male=>0, librarian => 1[统一映射]
因为需要针对这个人推荐，所以我这里找出他看过的电影以及评分记录：
19     274     2       879539794
也就是说，看过的电影id是274，
特征向量就是 x = [19,40,0,1,274,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0]


获取用户信息：
25|39|M|engineer|55107   => user id | age | gender | occupation | zip code
形成一个字典。{userId: {age;gender;occupation;zip_code;}}
"""
def getUserInfo(userInfoPath):
    userInfo = {}  # 初始化一个字典，用于存储用户的信息
    # 访问评分信息
    with open(userInfoPath) as file:  # 这句话是什么意思 => 打开filePathName所指的那个文件，然后将其存储在文件对象file中
        for line in file.readlines(): # 读取每行
            line = line.strip("\n") # 去掉行末的换行符  => 这个意思是： 每行的行末都有一个换行符也被读出来了
            # 为每个用户形成一个字典
            temp = extractUserInfo(line)
            userInfo[temp['id']] = temp
    return userInfo
    # print("=======用户字典如下========")
    # print(userInfo)
"""
1.从文件中获取数据放入Dataset中
"""


# 得到电影的数据
def getMovieInfo(movieInfoPath):
    movieInfo = {} # 电影字典
    with open(movieInfoPath,encoding = "ISO-8859-1") as file:
        for line in file.readlines():
            temp = {}  # 临时一个字典
            line = line.strip("\n")
            line = line.split("|") # 得到电影的信息数据
            temp['id'],temp['name'] = line[0] ,line[1]
            # 把topic 转换成一个int  temp[topic]的大小是19
            temp['topic'] =[int(_) for _ in  line[5:]]
            movieInfo[int(line[0])] = temp
    return movieInfo


#print(getMovieInfo("../data/ml-100k/u.item_exam"))