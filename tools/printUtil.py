"""
打印出配置信息
"""
def printConfig(argv):
    print("================ 参数配置如下：================")
    if argv[1] == 'train':
        print("当前执行的脚本文件是：", argv[0])
        print("训练数据：", argv[2])
        print("BATCH_SIZE =",argv[-3])  # =号后会自动跟一个空格
        print("EPOCH_TRAIN =",argv[-2])
    else:  # test
        print("当前执行的脚本文件是：", argv[0])
        print("测试数据：", argv[2])
        print("BATCH_SIZE =", argv[-3])
        print("EPOCH_TEST =", argv[-1])