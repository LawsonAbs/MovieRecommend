import queue as Q
"""
打印与用户ua最为相似的10个用户
"""
def printSimUser(ua):
    print("------------------------------")
    print("与用户", ua, "最为相似的前10个用户是：")
    while ua.simFriends.qsize() > 0 : # 当队列不为空时
        print(ua.simFriends.get()[1]) # 打印出其相似度

"""
# 打印推荐信息
"""
def printRecom(user):
    print("猜你喜欢：")
    for i in user.movies_rec:
        print(i,end=" ")


"""
计算用户的推荐
01.使用UserCF算法
02.TODO:应该去掉这个用户之前已经看过的电影集合
"""
def userCFRecom(user):
    # step 1.计算与用户user相似用户的相关电影
    while user.simFriends.qsize() > 0:  # 当队列不为空时
        cur = user.simFriends.get()
        friend = cur[1]  # 找出其相似好友
        poss = cur[0] # 其相似概率

        # 遍历相似用户看过的电影集合
        for i in friend.movies_read:
            val = user.movies_calc.get(i,0) # 初始值
            user.movies_calc[i] = val + (poss * friend.rateInfo.get(i))

    # step 2.将这些相关电影排序得到最后的输出
    user.movies_calc = sorted(user.movies_calc.items(),key=lambda x:x[1],reverse=True)

    print("为用户",user,"推荐的电影有：")
    for i in user.movies_calc[0:5]:
        print(i,end=" ")

"""
为用户推荐电影
01.使用ItemCF 算法，为用户 user 进行推荐
"""
def itemCF(user):
    pass