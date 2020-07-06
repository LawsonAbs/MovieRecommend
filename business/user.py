import queue as Q # 引入对列包

"""
用户类
"""
class User():
    # 定义构造方法
    def __init__(self, id, username, pasword):
        self.id = id
        self.username = username
        self.password = pasword
        self.rating={} # 评分
        self.simFriends = Q.PriorityQueue(11) # 相似好友的最大个数为10，但是存11个
        self.movies_read=[] # 已经看过的电影
        self.movies_calc = {} # 正在计算中的电影
#        self.movies_rec=Q.PriorityQueue(6) # 待推荐的5部电影

    def __str__(self) -> str:
        return self.username

    def __repr__(self) -> str:
        return self.username

    # 定义排序函数 =>【可能会有疑问，为什么在user中也有一个比较函数】
    def __lt__(self, other):
        return self.id < other.id