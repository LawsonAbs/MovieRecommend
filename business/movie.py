"""
商品类【在本案例中就是movie】
01.保存商品的信息，诸如【其和其它商品之间的相似度】
   因为要存储所有的商品间的相似信息，所以我用一个list 存储
"""
class Item():
    # itemId 表是商品id
    def __init__(self,itemId):
        self.itemId = itemId
        self.rateInfo = {} # 创建一个字典，存储这部电影的评分人及评分
        self.simItem = {}  #与本商品相似的商品及相似度

    def __str__(self):
        return "商品"+str(self.itemId)

    def __repr__(self):
        return "商品" + str(self.itemId)