## 数据集

数据集和算法目标详细说明请看[数据集说明](data%2Fserverless%2FREADME.md)

## 文件夹说明

`./papers`：存放我参考的毕业论文

`./data`: 数据集

`./checkpoints`: 存放训练好的模型

`utils`: 存放一些工具函数

`models`: 存放模型的定义

## 代码说明

`eda.ipynb`: 数据探索

`data_preprocess.ipynb`: 数据预处理

`CNN_LSTM_Attention.ipynb`: 基于CNN+LSTM+Attention的多变量时序预测模型

## 实验记录
### Seq2Seq多输出预测

训练参数：
```python
class Config:
    # data_path = './datasets/serverless/cached/queue_id_{}.csv'.format(36)
    data_path = './datasets/serverless/data.csv'
    timestep = 126  # 时间步长，就是利用多少时间窗口
    batch_size = 16  # 批次大小
    feature_size = 7  # 每个步长对应的特征数量（跟数据集处理有关，我只保留了七个特征）
    hidden_size = 256  # 隐层大小
    output_size = 1  # 只预测CPU
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
    lradj = 'type1'  # 学习率的调整方式，默认为"type1"
    loss_name = 'MSE'  # 损失函数名称 ['MSE', 'MAPE', 'MASE', 'SMAPE']
    scale_type = 'standard'  # 标准化类型 "standard" "minmax"
    save_path = './checkpoints/{}.pth'.format(model_name)  # 最优模型保存路径
```

测试集结果：

- mse: 34.302825927734375
- mae: 2.611034870147705
- rmse: 5.856861591339111
- mape: 1196964.0
- mspe: 70337874100224.0