import argparse

from exp.exp_seq2seq import Exp_Seq2Seq

parser = argparse.ArgumentParser(description='Seq2Seq time series Forecasting')

parser.add_argument('--mode', type=str, default='all',
                    help='mode of run, options: [all, train, test, pred]')
args = parser.parse_args()


class Config:
    data_path = './datasets/serverless/data.csv'
    timestep = 126  # 时间步长，就是利用多少时间窗口
    batch_size = 16  # 批次大小
    feature_size = 7  # 每个步长对应的特征数量（跟数据集处理有关，我只保留了七个特征）
    hidden_size = 256  # 隐层大小
    output_size = 1  # 只预测CPU
    pre_len = 24  # 预测长度
    num_layers = 2  # RNN的层数
    bidirectional = True
    epochs = 50  # 迭代轮数
    learning_rate = 0.001  # 学习率
    patience = 3  # 早停机制，如果损失多少个epochs没有改变就停止训练。
    model_name = 'seq2seq'  # 模型名称
    features = 'MS'  # 三个选项M，MS，S。分别是多元预测多元，多元预测单元，单元预测单元
    use_gpu = True
    gpu = 0
    checkpoints = './checkpoints/'
    inverse = False
    target = 'CPU_USAGE'  # 预测目标
    lradj = 'type1'  # 学习率的调整方式 ['type1', 'type2', 'cosine']
    loss_name = 'MSE'  # 损失函数名称 ['MSE', 'MAPE', 'MASE', 'SMAPE']
    scale_type = 'standard'  # 标准化类型 "standard" "minmax"
    save_path = './checkpoints/{}.pth'.format(model_name)  # 最优模型保存路径


config = Config()

# setting record of experiments
setting = 'group_id_{}_ft{}_ts{}_fs{}_os{}'.format(config.model_name, config.features,
                                                   config.timestep, config.feature_size, config.output_size)

exp = Exp_Seq2Seq(config)

if args.mode == 'all' or args.mode == 'train':
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

if args.mode == 'all' or args.mode == 'test':
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, load=True)

if args.mode == 'all' or args.mode == 'pred':
    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.predict(setting, load=True)
