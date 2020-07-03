import tools.util as ut
from algorithms import similarity as sim

userList = [] # 用于存放所有的User

if __name__ == '__main__':
    ut.readDataFromCsv("/Users/gamidev/program/resources/ml-25m/example.csv",userList)
    ut.printUserRate(userList)
    # 得到用户的相似度 => 注意这里的循环遍历方式
    for i in range(len(userList)):
        for j in range(i,len(userList)):
            u1 = userList[i]
            u2 = userList[j]
            print("用户",u1 ,"和用户",u2,"的相似度是：",sim.cosSimilarity(u1,u2))