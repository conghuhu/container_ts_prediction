import argparse

from exp.exp_transformer import Exp_Transformer

parser = argparse.ArgumentParser(description='Seq2Seq time series Forecasting')

parser.add_argument('--mode', type=str, default='all',
                    help='mode of run, options: [all, train, test, pred]')
args = parser.parse_args()


class Config:
    # data_path = './datasets/serverless/cached/queue_id_{}.csv'.format(36)
    data_path = './datasets/serverless/data.csv'
    timestep = 126  # 时间步长，就是利用多少时间窗口
    batch_size = 16  # 批次大小
    feature_size = 7  # 每个步长对应的特征数量（跟数据集处理有关，我只保留了七个特征）
    hidden_size = 256  # 隐层大小
    output_size = 7  # 预测变量数
    label_len = 64  # start token length
    pre_len = 24  # 预测长度
    num_layers = 1  # RNN的层数
    epochs = 50  # 迭代轮数
    learning_rate = 0.001  # 学习率
    patience = 5  # 早停机制，如果损失多少个epochs没有改变就停止训练。
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

    output_attention = False  # whether to output attention in ecoder
    enc_in = 7
    dec_in = 7
    c_out = output_size
    factor = 3
    d_model = 512  # dimension of model
    embed = 'timeF'
    freq = 'h'
    dropout = 0.1
    n_heads = 8  # num of heads
    e_layers = 2  # num of encoder layers
    d_layers = 1  # num of encoder layers
    d_ff = 2048  # dimension of fcn
    activation = 'gelu'  # activation


config = Config()

# setting record of experiments
setting = 'group_id_{}_ft{}_ts{}_fs{}_os{}'.format(config.model_name, config.features,
                                                   config.timestep, config.feature_size, config.output_size)
# TODO 未完成
exp = Exp_Transformer(config)

if args.mode == 'all' or args.mode == 'train':
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    # exp.train(setting)

if args.mode == 'all' or args.mode == 'test':
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.test(setting, load=True)

if args.mode == 'all' or args.mode == 'pred':
    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.predict(setting, load=True)
