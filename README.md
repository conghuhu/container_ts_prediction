## 数据集

数据集和算法目标详细说明请看[数据集说明](data%2Fserverless%2FREADME.md)

## 文件夹说明

`./papers`：存放我参考的毕业论文

`./datasets`: 数据集

`./checkpoints`: 存放训练好的模型

`./cached`: 缓存训练、测试数据的tensor

`./data`: dataloader，处理训练、测试数据的核心代码

`./exp`: 定义各种模型的train、test、pred方法，模型的入口。如果引入新模型，在此增加`exp_xxx.py`即可

`./predict_imgs`: 存放模型预测不同QUEUE_ID结果的图片

`./results`: 存放模型metrics结果

`./utils`: 存放一些工具函数

`./models`: 存放模型的定义

## 代码说明

`eda.ipynb`: 数据探索

`data_preprocess.ipynb`: 数据预处理

`CNN_LSTM_Attention.ipynb`: 基于CNN+LSTM+Attention的多变量时序预测模型

`main_seq2seq.py`: seq2seq模型训练、测试、预测入口类，直接运行即可

## 新增模型流程

1. 在`./models`下定义模型的`xxx.py`文件
2. 在`./exp`下定义模型的`exp_xxx.py`文件，继承`exp.Exp_Basic`
   类，实现`_build_model`、`_load_data`、`_get_data` 、`train`、`test`、`pred`、`vali`方法，可参考`exp/exp_seq2seq.py`。
3. 新建`main_xxx.py`，参考`main_seq2seq.py`定义好参数。
4. 训练模型：`python main_xxx.py --mode train`
5. 测试模型：`python main_xxx.py --mode test`
6. 预测模型：`python main_xxx.py --mode pred`

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

- mse: 0.5169850587844849
- mae: 0.2947041988372803
- rmse: 0.7190167307853699
- mape: 0.9166780114173889
- mspe: 7.035383701324463