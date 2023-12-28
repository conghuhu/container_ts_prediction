from exp.exp_seq2seq import Exp_Seq2Seq


class Config:
    data_path = './datasets/serverless/cached/queue_id_{}.csv'.format(36)
    timestep = 12  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    feature_size = 7  # 每个步长对应的特征数量（跟数据集处理有关，我只保留了七个特征）
    hidden_size = 256  # 隐层大小
    output_size = 6
    pre_len = output_size
    num_layers = 5  # lstm的层数
    epochs = 100  # 迭代轮数
    learning_rate = 0.0001  # 学习率
    patience = 10  # 早停机制，如果损失多少个epochs没有改变就停止训练。
    model_name = 'seq2seq'  # 模型名称
    features = 'MS'  # 三个选项M，MS，S。分别是多元预测多元，多元预测单元，单元预测单元
    use_gpu = True
    gpu = 0
    checkpoints = './checkpoints/'
    inverse = False
    target = 'CPU_USAGE'
    lradj = 'type1'  # 学习率的调整方式，默认为"type1"
    loss_name = 'MSE'
    scale_type = 'standard'  # 标准化类型 "standard" "minmax"
    save_path = './checkpoints/{}.pth'.format(model_name)  # 最优模型保存路径


config = Config()

# setting record of experiments
setting = 'group_id_{}_ft{}_ts{}_fs{}_os{}'.format(config.model_name, config.features,
                                                   config.timestep, config.feature_size, config.output_size)

exp = Exp_Seq2Seq(config)

print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
exp.train(setting)

print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
exp.test(setting)

print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
exp.predict(setting, load=True)
