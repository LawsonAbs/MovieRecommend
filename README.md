# 电影推荐  
本项目主要做如下几件事情：  
## 1.使用不同的推荐算法进行用户电影推荐，包括但不限于:
- UserCF
- ItemCF
- Logistic Regression

## 2.技术栈
- 推荐算法【最重要】 
- Django【尚未实现】
- git【一直在用】 

## 3.文件夹构成：
```
MovieRecommend
|--- __init__.py => 程序入口【暂未开启】
|
|
|----algorithms
|    是常用算法包。在其中会实现常用的算法如UserCF, ItemCF, Logistic Regression 等  
|    |---logisticRegression.py=> 用于实现逻辑回归算法
|    |---similarity.py        => 用于计算相似度的算法实现
|    |---recommend.py         => UserCF/ItemCF 算法实现
|
|----business
|    |----movie.py            => 电影信息数据
|    |----user.py             => 用户信息数据
|
|
|----data
|     data是资源包。本工程使用的电影评分数据就是放在这里，其下载路径是：http://files.grouplens.org/datasets/movielens
|
|----demo
|    算法示例，测试包
|
|
|----tools
     是工具包。用于读取数据，写入训练好的模型等；    
```
## 4.欢迎大家积极讨论，有问题可以直接提出Issues
