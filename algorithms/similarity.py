from tools import util as ut

"""
01.使用两种不同的相似度计算方法
"""

# 余弦相似度
# 传入的是俩个用户引用，计算其相似度
def cosSimiForUser(u1,u2):
    # 分别得到两个人的评分
    rate1 = u1.rateInfo
    rate2 = u2.rateInfo

    # 得到两个向量的长度，并将其相乘得到 multi2
    lenA = ut.getLenOfVector(rate1)
    lenB = ut.getLenOfVector(rate2)
    multi2 = lenA * lenB

    multi1 = 0 # 记录点乘的结果
    key1 = list(rate1.keys()) # 转换成键列表
    key2 = list(rate2.keys())
    i = 0
    j = 0
    lK1 = len(key1)
    lK2 = len(key2)

    while i < lK1 and j < lK2:
        # 在下面这个while循环中加入j<lK2 的原因是：避免j的下标溢出
        while i < lK1 and j < lK2 and key1[i] < key2[j]:
            i += 1
        while i < lK1 and j < lK2 and key2[j] < key1[i]:
            j += 1
        if i < lK1 and j < lK2 and key1[i] == key2[j]:
            multi1 += (rate1[key1[i]]) * (rate2[key2[j]])
            i += 1 # 再分别往后移动一位
            j += 1

    res = 0 # 预定义
    if multi2 != 0 : # 除数不能为0
        res = multi1/multi2 # 最后计算出的相似度
    print("用户", u1, "和用户", u2, "的点乘结果是：",multi1,"；模长乘积是：",multi2,"；余弦相似度是：",res)

    if u1.simFriends.qsize() > 10: # 弹出队列首部
        u1.simFriends.get()
    if u2.simFriends.qsize() > 10:
        u2.simFriends.get()

    if u1 is not u2: # 如果是同一个用户
        # 优先队列中
        u1.simFriends.put((res,u2))
        u2.simFriends.put((res,u1))
    return res


"""
01.为item计算余弦相似度
"""
def cosSimiForItem(i1,i2):
    # 分别得到两个人的评分
    rate1 = i1.rateInfo
    rate2 = i2.rateInfo

    # 得到两个向量的长度，并将其相乘得到 multi2
    lenA = ut.getLenOfVector(rate1)
    lenB = ut.getLenOfVector(rate2)
    multi2 = lenA * lenB

    multi1 = 0 # 记录点乘的结果
    key1 = list(rate1.keys()) # 转换成键列表
    key2 = list(rate2.keys())
    i = 0
    j = 0
    lK1 = len(key1)
    lK2 = len(key2)

    # 下面这么做的前提是，rateInfo 中的信息是有序的
    while i < lK1 and j < lK2:
        # 在下面这个while循环中加入j<lK2 的原因是：避免j的下标溢出
        while i < lK1 and j < lK2 and key1[i] < key2[j]:
            i += 1
        while i < lK1 and j < lK2 and key2[j] < key1[i]:
            j += 1
        if i < lK1 and j < lK2 and key1[i] == key2[j]:
            multi1 += (rate1[key1[i]]) * (rate2[key2[j]])
            i += 1 # 再分别往后移动一位
            j += 1

    res = 0 # 预定义
    if multi2 != 0 : # 除数不能为0
        res = multi1/multi2 # 最后计算出的相似度
    print("%s" %i1, "和%s" %i2, "的点乘结果是：%f" %multi1,"；模长乘积是：",multi2,"；余弦相似度是：",res)
    i1.simItem.setdefault(i2.itemId,res) # 添加相似度信息
    i2.simItem.setdefault(i1.itemId,res)  # 添加相似度信息
    return res

# 皮尔逊相似度
def pearsonSimilarity(self):
    pass