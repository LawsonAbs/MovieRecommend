import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)  # 手动设置一个随机数，用于产生随机数

# Softmax is also in torch.nn.functional
data = torch.randn(5)  # 生成一个 "list" 含5个元素。实际上其类型是 <class 'torch.Tensor'>
print(data)
# 将 softemax() 函数应用到data的第dim维中。 这里的维数我还不是特别清楚
print(F.softmax(data, dim=0))  # 这里的代码是存在bug的，因为上面申述的data 只有一维，引用下标dim=1显然有问题
print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!  => 可以看到这里最后得到的值为1
print(F.log_softmax(data, dim=0))  # theres also log_softmax  => 功能等价于log(softmax(x)) 但如果分开实现是慢于 log_softmax(x)的。

# 初始化一波数据，这是明显的list中嵌套tuple； 下面的test_data 也是相同的道理
data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("it is lost on me".split(), "ENGLISH"),("Yo creo que si".split(), "SPANISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:  # 直接用+号将两个list连接
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}


class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module        
        # super(BoWClassifier,self) 首先找到 BoWClassifier 的父类（就是类 nn.Module），然后把类 BoWClassifier 的对象转换为类 nn.module 的对象
        super(BoWClassifier, self).__init__()  # super()函数的第一个参数实际意义不大【这个应该算是Python2的语法】

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    # 定义forward()函数的作用是什么？
    # 这个方法是所有继承nn.Module 类都应该实现的方法
    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=1)


"""解释一下参数
sentence:传入的是一个个的句子。在本案例中就是：  me gusta comer en la cafeteria 这种句子
"""
def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))  # 搞成一个零向量，大小为word_to_ix 的长度
    # 这里用
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


# 实例化BoWClassifier
model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# the model knows its parameters.  The first output below is A, the second is b.
# Whenever you assign a component to a class variable in the __init__ function
# of a module, which was done with the line
# self.linear = nn.Linear(...)
# Then through some Python magic from the PyTorch devs, your module
# (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
"""
上面话的意思是：下面这个for循环会输出两个参数变量，分别是A 和 b。它们是y = Ax+b 中的两个变量
"""
for param in model.parameters():
    print(param.size()) # A.size()=[2,26]   b.size()=[2]
    print(param)

# To run the model, pass in a BoW vector
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad(): # 使用不求导
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)   # 这是什么操作？ 为何实例也可以当做方法调用？
    print(log_probs)

# Run on demo data before we train, just to see a before-and-after
print("-------before-----------")
with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# Print the matrix column corresponding to "creo"
print(next(model.parameters())[:, word_to_ix["creo"]])

loss_function = nn.NLLLoss() # 定义损失函数
optimizer = optim.SGD(model.parameters(), lr=0.1) # 使用SGD算法进行优化，学习率为0.1

# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
for epoch in range(100):
    for instance, label in data:
        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Tensor as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step() # 执行一次更新参数的操作

print("---------after-----------")
# 为何这里还是用 torch.no_grad() 函数？
with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# Index corresponding to Spanish goes up, English goes down!
#print(next(model.parameters())[:, word_to_ix["creo"]])
