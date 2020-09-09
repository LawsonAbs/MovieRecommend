"""
功能：读取json文件，其读取到的内容类型是list，并将制定字段内容返回成字典的集合
参数：
    01.filePath: json文件的地址
    02.返回一个list，数据格式如下
    [[user_id,role,age,travel_list,income]
    ...
    ]
    其中travel_list 是list中的数据，
"""
import json

# 根据标签返回该标签的id。用户兴趣标签
interest2Tag = {'makeups':0,"skincare":1,"food":2,
            "mother-baby":3,"outfit":4,"travel":5,
            "parenting":6,"read":7,"sport":8,"game":9,
            "pet":10,"film":11,"school":12,"car":13,
            "digital":14,"other":15 }

# 收入标签，年龄标签，性别标签，角色标签
income2Tag ={'lt-5000':0,'5000-8000':1,'8000-10000':2,'10000-20000':3,'gt-20000':4}
age2Tag={'lt-18':0,'18-24':1,'25-30':2,'30-37':3,'38-50':4,'gt-50':5}
gender2Tag={'male':0,'female':1}
role2Tag={'whitecollar':0,'student':1,'professional':2,'home_mom':3}

# 商品品类标签  => 有12个
class2Tag={'11000001':0,'11000002':1,'11000003':2,
           '11000004':3,'11000005':6,'11000006':5,
           '11000007':6,'11000008':7,'11000009':8,
           '11000010':9,'11000011':10,'11000012':11,}

app2Tag={"android":0,"ios":1}

"""
将product 的相关文件转为dic，这个dic的特征是 product 到商品类别的映射
brand_id -> classifyTag
"""
def readProductJson2Dict(filePath):
    productClassTag = {}  # brand_id -> classify Tag
    with open(filePath,encoding="utf-8") as file:
        cont = json.load(file)
        for line in cont:
            brandId = line.get("brand_id")
            classifyId = line.get("classify_id")
            productClassTag[brandId] = class2Tag.get(classifyId)
    return productClassTag

def readUserJson2Dict(filePath):
    """将用户的数据转换成字典的形式
    {user_id:}
    :param filePath:
    :return: userInfo
    """
    userInfo = {}
    with open(filePath) as file:
        cont = json.load(file)  # 读取内容 => 类型为list
        for line in cont: # 内容中的每行
            if (len(line)) <= 1:  # 过滤掉不符合条件的数据
                continue
            tempDict = []  # 临时的list
            # 处理数据，分别得到每个状态下的数据
            if line.get("_id") is None or len(line.get("_id")) <=0:
                continue
            if line.get("archives") is None:
                continue
            user_id = line.get("_id")
            role = line.get("archives").get("role").replace("role:","")  # 去掉前面的role:
            age = line.get("archives").get("age").replace("age:","")
            interest_label = line.get("archives").get("interest_label")
            interest_label = [_.replace("interest:","") for _ in interest_label]
            income = line.get("archives").get("income").replace("income:","")

            # 分别处理上述得到的数据，得到数值向量
            ageId = age2Tag.get(age)
            roleId = role2Tag.get(role)
            interestId = [0 for _ in range(15) ]
            for _ in interest_label:
                interestId[interest2Tag.get(_)] = 1
            incomeId = income2Tag.get(income)

            # 设置字典值
            tempDict.append(roleId)
            tempDict.append(ageId)
            tempDict.extend(interestId)
            tempDict.append(incomeId)
            userInfo[user_id] = tempDict
    return userInfo

# 读取dispatch 的数据，返回该dispatch 的类型向量
def readDispatch2Dict(filePath,productClassTag):
    res = {} # 返回各个 dispatch 对应的向量信息
    with open(filePath,encoding='utf-8') as file:
        cont = json.load(file)
        productList = cont.get("data").get("list")
        #print(len(productList))
        for product in productList: # 打印每个产品
            temp = []
            dispatchId = product.get("dispatch_id")
            praiseCount = product.get("statistics").get("praise_count")
            shareCount = product.get("statistics").get("share_count")
            userCount = product.get("statistics").get("user_count")

            content = product.get("gift_bag").get("content")
            brandIdList = []
            # 生成1*12维大小的向量，这个用于代表dispatch 中的商品信息
            # 这么做的原因是，想用商品品类标签（12个）来衡量该dispatch 的特征
            vec = [0 for _ in range(12)]
            for cont in content:
                brandId = cont.get("brand_id")
                brandIdList.append(brandId)  # 得到dispatch 中的 brand_id 的集合
                index = productClassTag.get(brandId)  # 得到该brandId对应的classTag
                vec[index] = 1  # 找到相应的id进行修改

            temp.extend([praiseCount, shareCount, userCount])
            temp.extend(vec)  # 加上标签属性
            res[dispatchId] = temp
        return res


"""读取dispatch 和用户交互的信息
track.json 不是标准的json文件，所以这里是逐行读取内容，然后再用json格式分析
"""
def readTrackJson2Dict(filePath):
    """interact 的数据形式：
    {   user_id:{app_client:value,dispatch_id:duration,...},
        user_id:{app_client:value,dispatch_id:duration,...},
        ....
    }
    :param filePath:
    :return:
    """
    interact =  {} # 用户与产品的交互数据
    with open(filePath,encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip("\n") # 去行末空格
            cont = json.loads(line) # 使用json读取line
            #print(cont)
            duration = cont.get("duration") # get value or None
            # 去掉以下三种情况的duration
            if duration == "" or duration is None or duration == 0:
                continue
            user_id = cont.get("user_id")
            if user_id == "" or user_id is None:
                continue
            dispatch_id = cont.get("dispatch_id")
            app_client = cont.get("app_client")

            # 中间的一个临时字典
            tempDic = {}
            #tempDic["app_client"] = app_client
            tempDic[dispatch_id] = duration

            # 累积对每个dispatch_id 的浏览时长
            if user_id in interact.keys(): # 如果之前就已经添加过了
                old = interact.get(user_id).get(dispatch_id, 0)  # 入股没有则为0
                new = old + duration
                interact[user_id].setdefault(dispatch_id, new)
            else: # 如果之前没有添加过
                interact[user_id] = tempDic
                # 对字典的值进行修改
                interact.get(user_id).setdefault(dispatch_id,duration)
    return interact


"""得到一个完整的数据集
根据每条交互信息生成一条特征向量，这条特征向量就是用于训练模型的数据。
"""
def getWholeData(userInfo,interact,dispatchInfo):
    feature = []  # 特征数据
    label = []  # 标签数据
    for item in interact.items():
        key = item[0]
        val = item[1]
        cur = []  # 表示当前这条数据(18)
        if userInfo.get(key) is None:
            continue
        cur.extend(userInfo.get(key))
        for i in val.items():  # 遍历各个dispatch_id
            temp = [_ for _ in cur]
            dispatchId = i[0]
            duration = i[1]
            if dispatchInfo.get(dispatchId) is None:
                continue
            temp.extend(dispatchInfo.get(dispatchId))  # 放入特征(33)
            feature.append(temp)  # 放入结果特征集中
            if duration >= 100: # 说明浏览时长够长=>推荐购买
                label.append(1)
            else : # 不推荐
                label.append(0)
    return feature,label
