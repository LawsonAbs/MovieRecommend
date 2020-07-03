"""
用户类
"""
class User():
    #定义构造方法
    def __init__(self,id,username,pasword):
        self.id = id
        self.username = username
        self.password = pasword

    def __str__(self) -> str:
        return self.username

    def __repr__(self) -> str:
        return self.username


