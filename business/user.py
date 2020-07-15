import queue as Q # 引入队列包

"""
用户类
"""
class User():
    # 定义构造方法
    def __init__(self, id, username, pasword):
        self.id = id
        self.username = username # 用户名
        self.password = pasword # 用户密码，暂未用到
        self.rateInfo={} # 评分信息。类似于{movie:rate}
        self.simFriends = Q.PriorityQueue(11) # 相似好友的最大个数为10，但是存11个
        self.moviesRead=[] # 已经看过的电影
        self.moviesCalc = {} # 正在计算中的电影，存储形式是{movie:posibility}  => 最后的结果就是待推荐的电影

    def __str__(self) -> str:
        return self.username

    def __repr__(self) -> str:
        return self.username

    # 定义排序函数 =>【可能会有疑问，为什么在user中也有一个比较函数】
    # 这个函数在放入优先队列中会使用到，如果用户相似度值相同，则按照id的大小顺序排序入队
    def __lt__(self, other):
        return self.id < other.id