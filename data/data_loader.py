import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset

from utils.tools import torch_to_csv, csv_to_torch


class Dataset_Custom(Dataset):
    def __init__(self, args, data_path, size, flag='train',
                 features='S',
                 target='CPU_USAGE', ratio=0.8, scale=True, scale_type='standard', inverse=False, cols=None):
        # size [timestep, pred_len]
        # info
        self.timestep = size[0]
        self.feature_size = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test']
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.cols = cols
        self.data_path = data_path
        self.ratio = ratio
        assert scale_type in ['standard', 'minmax']
        self.scale_type = scale_type
        self.args = args
        self.__read_data__()

    def __read_data__(self):
        # 归一化器
        if self.scale_type == 'standard':
            self.scaler = StandardScaler()
            self.scaler_model = StandardScaler()
        elif self.scale_type == 'minmax':
            self.scaler = MinMaxScaler()
            self.scaler_model = MinMaxScaler()
        # 读取数据并预处理
        # 默认第一列时间戳为index
        df_raw = pd.read_csv(self.data_path, index_col=0)
        print("读取到本地数据： \n", df_raw.head())

        '''
        df_raw.columns: ['timestamp(index)', target feature, ...(other features)]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
        df_raw = df_raw[[self.target] + cols]

        if self.features == 'S':
            # 单元预测，只需要一个元
            df_data = df_raw[[self.target]]
        else:
            df_data = df_raw

        if self.scale:
            data: np.ndarray = self.scaler_model.fit_transform(df_data.values)
            self.scaler.fit_transform(np.array(df_data[self.target]).reshape(-1, 1))
        else:
            data: np.ndarray = df_data.values

        # 划分训练集和测试集
        self.__split_data__(data, self.timestep, self.feature_size, self.pred_len)

        print("x_train shape: ", self.x_train.shape)
        print("y_train shape: ", self.y_train.shape)
        print("x_test shape: ", self.x_test.shape)
        print("y_test shape: ", self.y_test.shape)

        if self.flag == 'train':
            self.data_x = self.x_train
            self.data_y = self.y_train
            # 将数据保存到本地
            torch_to_csv(self.x_train, './cached/x_train_{}.pt'.format(self.args.model_name))
            torch_to_csv(self.y_train, './cached/y_train_{}.pt'.format(self.args.model_name))
        else:
            self.data_x = self.x_test
            self.data_y = self.y_test
            torch_to_csv(self.x_test, './cached/x_test_{}.pt'.format(self.args.model_name))
            torch_to_csv(self.y_test, './cached/y_test_{}.pt'.format(self.args.model_name))

    def __split_data__(self, data: np.ndarray, timestep: int, feature_size: int, output_size: int):
        print(data.shape, timestep, feature_size, output_size)
        """
        形成训练数据，例如12345789 12-3456789
        :param data: 数据
        :param timestep: 历史时间步的长度
        :param feature_size: 特征数
        :return:
        """
        dataX = []  # 保存X
        dataY = []  # 保存Y

        # 将整个窗口的数据保存到X中，将未来一天保存到Y中
        for index in range(len(data) - timestep - output_size + 1):
            # 第一列是Target, CPU_USAGE
            dataX.append(data[index: index + timestep])
            dataY.append(data[index + timestep: index + timestep + output_size][:, 0].tolist())

        dataX = np.array(dataX)
        dataY = np.array(dataY)

        # 获取训练集大小
        train_size = int(np.round(self.ratio * dataX.shape[0]))

        # 划分训练集、测试集
        x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
        y_train = dataY[: train_size].reshape(-1, output_size)

        x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
        y_test = dataY[train_size:].reshape(-1, output_size)

        # 将数据转为tensor
        x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
        y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
        x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
        y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

        self.x_train = x_train_tensor
        self.y_train = y_train_tensor
        self.x_test = x_test_tensor
        self.y_test = y_test_tensor

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler_model.inverse_transform(data)

    def inverse_transform_y(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, args, data_path, size, flag='pred',
                 features='MS', target='CPU_USAGE', ratio=0.8, scale=True, scale_type='standard', inverse=False,
                 cols=None):
        # size [seq_len, label_len, pred_len]
        # info

        self.timestep = size[0]
        self.feature_size = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.cols = cols
        self.data_path = data_path
        self.ratio = ratio
        assert scale_type in ['standard', 'minmax']
        self.scale_type = scale_type
        self.args = args
        self.__read_data__()

    def __read_data__(self):
        # 归一化器
        if self.scale_type == 'standard':
            self.scaler = StandardScaler()
            self.scaler_model = StandardScaler()
        elif self.scale_type == 'minmax':
            self.scaler = MinMaxScaler()
            self.scaler_model = MinMaxScaler()
        # 默认第一列时间戳为index
        df_raw = pd.read_csv(self.data_path, index_col=0)
        print("读取到本地数据： \n", df_raw.head())
        '''
        df_raw.columns: ['timestamp(index)', target feature, ...(other features), ]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
        df_raw = df_raw[[self.target] + cols]

        if self.features == 'S':
            # 单元预测，只需要一个元
            df_data = df_raw[[self.target]]
        else:
            df_data = df_raw

        if self.scale:
            data: np.ndarray = self.scaler_model.fit_transform(df_data.values)
            self.scaler.fit_transform(np.array(df_data[self.target]).reshape(-1, 1))
        else:
            data: np.ndarray = df_data.values

        self.x_test = csv_to_torch('./cached/x_test_{}.pt'.format(self.args.model_name))
        self.y_test = csv_to_torch('./cached/y_test_{}.pt'.format(self.args.model_name))

        print("x_test shape: ", self.x_test.shape)
        print("y_test shape: ", self.y_test.shape)

        self.data_x = self.x_test
        self.data_y = self.y_test

        # self.data_x = data[border1:border2]
        # if self.inverse:
        #     self.data_y = df_data.values[border1:border2]
        # else:
        #     self.data_y = data[border1:border2]

    def __getitem__(self, index):
        n = self.__len__()
        return self.data_x[n - 1], self.data_y[n - 1]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        if torch.is_tensor(data):
            data = data.cpu().detach().numpy().reshape(-1, 1)
        return self.scaler_model.inverse_transform(data)

    def inverse_transform_y(self, data):
        if torch.is_tensor(data):
            data = data.cpu().detach().numpy().reshape(-1, 1)
        return self.scaler.inverse_transform(data)
