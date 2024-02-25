import argparse

from huawei.exp.exp_dlinear import Exp_DLinear

parser = argparse.ArgumentParser(description='CNN_LSTM_Attention time series Forecasting')

parser.add_argument('--mode', type=str, default='all',
                    help='mode of run, options: [all, train, test, pred]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoints path')
parser.add_argument('--run_type', type=str, default='shell', help='run type, options: [shell, ide]')
args = parser.parse_args()


class Config:
    # basic config
    model_name = 'DLinear'  # 模型名称
    save_path = './checkpoints/huawei/{}.pth'.format(model_name)  # 最优模型保存路径

    # data loader
    data_path = './datasets/huawei/data.csv'
    features = 'MS'  # 三个选项M，MS，S。分别是多元预测多元，多元预测单元，单元预测单元
    target = 'total_cpu_usage'  # 预测目标
    checkpoints = args.checkpoints
    scale_type = 'standard'  # 标准化类型 "standard" "minmax"

    # forecasting task
    timestep = 144  # 时间步长，就是利用多少时间窗口
    feature_size = 18  # 每个步长对应的特征数量
    pre_len = 24  # 预测长度
    inverse = False

    # model define
    enc_inc = feature_size  # encoder input size
    moving_avg = 25  # 移动平均窗口
    individual = False  # 针对DLinear是否为每个变量（通道）单独建立一个线性层
    use_RevIN = False

    # optimization
    epochs = 100  # 迭代轮数
    batch_size = 256  # 批次大小
    patience = 5  # 早停机制，如果损失多少个epochs没有改变就停止训练。
    learning_rate = 0.001  # 学习率
    loss_name = 'MSE'  # 损失函数名称 ['MSE', 'MAPE', 'MASE', 'SMAPE', 'smoothl1']
    lradj = 'cosine'  # 学习率的调整方式 ['type1', 'type2', 'cosine']

    # GPU
    use_gpu = True
    gpu = 0

    pred_mode = 'paper'  # 预测模式 ['paper', 'show']
    test_show = 'brief'  # 测试集展示 ['all', 'brief']

    run_type = args.run_type  # 运行模式 ['shell', 'ide']， shell模式不show图片


config = Config()

# setting record of experiments
setting = 'huawei_{}_ft{}_ts{}_fs{}_pl{}_epoch{}_lr{}_bs{}_enc{}_mavg{}_individual{}_loss{}_revin{}'.format(
    config.model_name,
    config.features,
    config.timestep,
    config.feature_size,
    config.pre_len,
    config.epochs,
    config.learning_rate,
    config.batch_size,
    config.enc_inc,
    config.moving_avg,
    config.individual,
    config.loss_name, config.use_RevIN)

config.setting = setting
exp = Exp_DLinear(config)

if args.mode == 'all' or args.mode == 'train':
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

if args.mode == 'all' or args.mode == 'test':
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, load=True)

if args.mode == 'all' or args.mode == 'pred':
    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.predict(setting, load=True)
