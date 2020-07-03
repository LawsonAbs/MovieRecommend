from tools import util as ut

"""
01.使用两种不同的相似度计算方法
"""

# 余弦相似度
# 传入的是俩个用户引用，计算其相似度
def cosSimilarity(u1,u2):
    rate1 = u1.rating
    rate2 = u2.rating #分别得到两个人的评分
    # 得到两个向量的长度
    lenA = ut.getLenOfVector(rate1)
    lenB = ut.getLenOfVector(rate2)
    multi2 = lenA * lenB
    multi1 = 0 # 计算向量的点乘
    maxKey1 = max(rate1.keys())
    maxKey2 = max(rate2.keys())
    for i in range(max(maxKey1,maxKey2)):
        multi1 += (rate1.get(i,0)) * (rate2.get(i,0))
    print("点乘结果：",multi1)
    print("模长乘积：",multi2)
    res = multi1/multi2 # 最后计算出的相似度
    return res

# 皮尔逊相似度
def pearsonSimilarity(self):
    pass