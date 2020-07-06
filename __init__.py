import tools.util as ut
from algorithms import similarity as sim
from business import recommend as rec

userList = [] # 用于存放所有的User

if __name__ == '__main__':
    ut.readDataFromCsv("/Users/gamidev/program/resources/ml-25m/example.csv",userList)
    #ut.printUserRate(userList)
    # 得到用户的相似度 => 注意这里的循环遍历方式，因为想避开重复的相似度计算，所以不是朴素的双层for循环
    for i in range(len(userList)):
        for j in range(i,len(userList)):
            u1 = userList[i]
            u2 = userList[j]
            sim.cosSimilarity(u1,u2)

    # for i in userList:
    #     rec.printSimUser(i)

    ut.printInfo(userList)
    # 根据inputId 得到用户
    rec.calcRecom(userList[0])
