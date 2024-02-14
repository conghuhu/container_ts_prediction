import argparse

from exp.exp_dsformer import Exp_DsFormer

parser = argparse.ArgumentParser(description='DsFormer time series Forecasting')

parser.add_argument('--mode', type=str, default='all',
                    help='mode of run, options: [all, train, test, pred]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoints path')
parser.add_argument('--run_type', type=str, default='shell', help='run type, options: [shell, ide]')
args = parser.parse_args()


class Config:
    # basic config
    model_name = 'DsFormer'  # 模型名称
    save_path = './checkpoints/{}.pth'.format(model_name)  # 最优模型保存路径

    # data loader
    data_path = './datasets/serverless/data.csv'
    features = 'MS'  # 三个选项M，MS，S。分别是多元预测多元，多元预测单元，单元预测单元
    target = 'CPU_USAGE'  # 预测目标
    checkpoints = args.checkpoints
    scale_type = 'standard'  # 标准化类型 "standard" "minmax"

    # forecasting task
    timestep = 24  # 时间步长，就是利用多少时间窗口
    output_size = 1  # 只预测CPU
    feature_size = 8  # 每个步长对应的特征数量（跟数据集处理有关，我只保留了七个特征）
    pre_len = 24  # 预测长度
    inverse = False

    # model define
    hidden_size = 64  # 隐层大小
    num_layers = 3  # encoder层数
    num_heads = 2  # nhead数和d_model也就是嵌入维度必须满足整除关系
    ffn_hidden_size = 64
    queue_embed_dim = 100  # QUEUE_ID嵌入向量的维度
    dropout = 0.05

    # optimization
    epochs = 5  # 迭代轮数
    batch_size = 256  # 批次大小
    patience = 5  # 早停机制，如果损失多少个epochs没有改变就停止训练。
    learning_rate = 0.001  # 学习率
    loss_name = 'MSE'  # 损失函数名称 ['MSE', 'MAPE', 'MASE', 'SMAPE']
    lradj = 'cosine'  # 学习率的调整方式 ['type1', 'type2', 'cosine']

    # GPU
    use_gpu = True
    gpu = 0

    train_range = 'all'  # 训练集的范围 ['all', 'train']
    pred_mode = 'paper'  # 预测模式 ['paper', 'show']
    test_show = 'brief'  # 测试集展示 ['all', 'brief']

    run_type = args.run_type


config = Config()

# setting record of experiments
setting = 'group_id_{}_ft{}_ts{}_fs{}_os{}_pl{}_epoch{}_lr{}_bs{}_hs{}_rl{}_nh{}_dp{}_tr{}'.format(config.model_name,
                                                                                                   config.features,
                                                                                                   config.timestep,
                                                                                                   config.feature_size,
                                                                                                   config.output_size,
                                                                                                   config.pre_len,
                                                                                                   config.epochs,
                                                                                                   config.learning_rate,
                                                                                                   config.batch_size,
                                                                                                   config.hidden_size,
                                                                                                   config.num_layers,
                                                                                                   config.num_heads,
                                                                                                   config.dropout,
                                                                                                   config.train_range)
config.setting = setting
exp = Exp_DsFormer(config)

# if args.mode == 'all' or args.mode == 'train':
#     print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
#     exp.train(setting)
#
# if args.mode == 'all' or args.mode == 'test':
#     print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#     exp.test(setting, load=True)
#
# if args.mode == 'all' or args.mode == 'pred':
#     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#     exp.predict(setting, load=True)
