import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset

from utils.tools import csv_to_torch, torch_to_csv


class Dataset_Custom(Dataset):
    def __init__(self, args, data_path, size, flag='train',
                 features='S',
                 target='CPU_USAGE', ratio=0.8, scale=True, scale_type='standard', inverse=False, cols=None,
                 embed='timeF',
                 freq='min'):
        # size [timestep, pred_len]
        # info
        self.timestep = size[0]
        self.feature_size = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'all']
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

        timeenc = 0 if embed != 'timeF' else 1
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()

    def __read_data__(self):
        # 归一化器
        if self.scale_type == 'standard':
            self.scaler_y = StandardScaler()
            self.scaler_x = StandardScaler()
        elif self.scale_type == 'minmax':
            self.scaler_y = MinMaxScaler()
            self.scaler_x = MinMaxScaler()
        # 读取数据并预处理
        # 默认第一列时间戳为index
        data_df = pd.read_csv(self.data_path)
        # print("读取到本地csv数据： \n", data_df.head())
        print("加载{}数据集...".format(self.flag))

        # 时间特征编码
        # df_stamp = pd.DataFrame(data_df.index, columns=['timestamp'])
        # df_stamp.rename(columns={'timestamp': 'date'}, inplace=True)
        # df_stamp['date'] = pd.to_datetime(df_stamp.date, unit='ms')
        # print("df_stamp: \n", df_stamp)
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(labels=['date'], axis=1).values
        #     # data_stamp shape: [len, 4]
        # elif self.timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)
        #     # data_stamp shape: [len, 5]

        '''
        data_df.columns: ['timestamp(index)', target feature, ...(other features)]
        '''
        # cols = list(data_df.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(data_df.columns)
            cols.remove(self.target)
            cols.remove('QUEUE_ID')
        data_df = data_df[[self.target] + cols + ['QUEUE_ID']]

        if self.features == 'S':
            # 单元预测，只需要一个元
            df_data = data_df[[self.target]]
        else:
            df_data = data_df

        if self.scale:
            data: np.ndarray = self.scaler_x.fit_transform(df_data.values)
            self.scaler_y.fit_transform(np.array(df_data[self.target]).reshape(-1, 1))
        else:
            data: np.ndarray = df_data.values

        cache_tensor_path = os.path.join('./cached',
                                         'ts{}_fs{}_pl{}'.format(self.timestep, self.feature_size, self.pred_len))
        if not os.path.exists(cache_tensor_path):
            os.makedirs(cache_tensor_path)

        # 检查本地结果文件是否存在，如果存在直接返回
        x_train_cache_tensor_path = cache_tensor_path + '/x_train_{}.pt'.format(self.flag)
        y_train_cache_tensor_path = cache_tensor_path + '/y_train_{}.pt'.format(self.flag)
        x_test_cache_tensor_path = cache_tensor_path + '/x_test.pt'
        y_test_cache_tensor_path = cache_tensor_path + '/y_test.pt'
        if self.flag == 'train' or self.flag == 'all':
            if os.path.exists(x_train_cache_tensor_path) and os.path.exists(y_train_cache_tensor_path):
                self.data_x = csv_to_torch(x_train_cache_tensor_path)
                self.data_y = csv_to_torch(y_train_cache_tensor_path)
                print("读取本地训练集缓存数据： \n", self.data_x.shape, self.data_y.shape)
                return
        else:
            if os.path.exists(x_test_cache_tensor_path) and os.path.exists(y_test_cache_tensor_path):
                self.data_x = csv_to_torch(x_test_cache_tensor_path)
                self.data_y = csv_to_torch(y_test_cache_tensor_path)
                print("读取本地测试集缓存数据： \n", self.data_x.shape, self.data_y.shape)
                return

        x_train_tensor_list = []
        y_train_tensor_list = []
        x_test_tensor_list = []
        y_test_tensor_list = []

        test_start = 0
        test_starts = []
        test_ends = []
        queueIds_df = pd.read_csv('./datasets/serverless/q_ids.csv')
        queueIds: np.ndarray = queueIds_df['QUEUE_ID'].values
        for i, queueId in np.ndenumerate(queueIds):
            # i是只有一个下标index的tuple，访问时直接i[0]
            raw = queueIds_df.iloc[i[0]]
            start = raw['ranges_start']
            end = raw['ranges_end']
            # 划分训练集和测试集
            x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_all_tensor, y_all_tensor = self.__split_data__(
                data[start:end + 1],
                self.timestep,
                self.feature_size,
                self.pred_len)
            if self.flag == 'all':
                x_train_tensor_list.append(x_all_tensor)
                y_train_tensor_list.append(y_all_tensor)
            else:
                x_train_tensor_list.append(x_train_tensor)
                y_train_tensor_list.append(y_train_tensor)
            x_test_tensor_list.append(x_test_tensor)
            y_test_tensor_list.append(y_test_tensor)

            test_end = test_start + len(x_test_tensor) - 1
            test_starts.append(test_start)
            test_ends.append(test_end)
            test_start = test_end + 1

        x_train_tensor = torch.cat(x_train_tensor_list, dim=0)
        y_train_tensor = torch.cat(y_train_tensor_list, dim=0)
        x_test_tensor = torch.cat(x_test_tensor_list, dim=0)
        y_test_tensor = torch.cat(y_test_tensor_list, dim=0)
        print("x_train shape: ", x_train_tensor.shape)
        print("y_train shape: ", y_train_tensor.shape)
        print("x_test shape: ", x_test_tensor.shape)
        print("y_test shape: ", y_test_tensor.shape)

        queueIds_df['test_start'] = test_starts
        queueIds_df['test_end'] = test_ends
        queueIds_df.to_csv('./datasets/serverless/q_ids.csv', encoding="utf-8", index=False)

        if self.flag == 'train' or self.flag == 'all':
            self.data_x = x_train_tensor
            self.data_y = y_train_tensor
        else:
            self.data_x = x_test_tensor
            self.data_y = y_test_tensor
        # 将数据保存到本地
        torch_to_csv(x_train_tensor, x_train_cache_tensor_path)
        torch_to_csv(y_train_tensor, y_train_cache_tensor_path)
        torch_to_csv(x_test_tensor, x_test_cache_tensor_path)
        torch_to_csv(y_test_tensor, y_test_cache_tensor_path)

    def __split_data__(self, data: np.ndarray, timestep: int, feature_size: int,
                       pred_len: int):
        """
        形成训练数据，例如12345789 123456->789
        :param data: 数据
        :param timestep: 历史时间步的长度
        :param feature_size: 特征数
        :return:
        """
        dataX = []  # 保存X
        dataY = []  # 保存Y
        # print(data.shape, timestep, feature_size, pred_len)

        # 将整个窗口的数据保存到X中，将未来一天保存到Y中
        for index in range(len(data) - timestep - pred_len + 1):
            # 第一列是Target, CPU_USAGE
            dataX.append(data[index: index + timestep])
            dataY.append(data[index + timestep: index + timestep + pred_len][:, 0].tolist())

        dataX = np.array(dataX)
        dataY = np.array(dataY)

        # 获取训练集大小
        train_size = int(np.round(self.ratio * dataX.shape[0]))

        # 划分训练集、测试集
        x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
        y_train = dataY[: train_size].reshape(-1, pred_len, 1)

        x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
        y_test = dataY[train_size:].reshape(-1, pred_len, 1)

        # 将数据转为tensor
        x_train_tensor: torch.Tensor = torch.from_numpy(x_train).to(torch.float32)
        y_train_tensor: torch.Tensor = torch.from_numpy(y_train).to(torch.float32)
        x_test_tensor: torch.Tensor = torch.from_numpy(x_test).to(torch.float32)
        y_test_tensor: torch.Tensor = torch.from_numpy(y_test).to(torch.float32)

        x_all_tensor: torch.Tensor = torch.from_numpy(dataX).to(torch.float32)
        y_all_tensor: torch.Tensor = torch.from_numpy(dataY.reshape(-1, pred_len, 1)).to(torch.float32)

        return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_all_tensor, y_all_tensor

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler_x.inverse_transform(data)

    def inverse_transform_y(self, data):
        return self.scaler_y.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, args, data_path, size, dataset_obj: Dataset, flag='pred',
                 features='MS', target='CPU_USAGE', scale=True, inverse=False,
                 cols=None):
        # size [seq_len, label_len, pred_len]
        # info

        self.timestep = size[0]
        self.feature_size = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['pred']
        self.dataset_obj = dataset_obj

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.cols = cols
        self.data_path = data_path
        self.args = args
        self.__read_data__()

    def __read_data__(self):
        queueIds_df = pd.read_csv('./datasets/serverless/q_ids.csv')
        queueIds: np.ndarray = queueIds_df['QUEUE_ID'].values
        self.queueIds_df = queueIds_df
        self.queueIds = queueIds

        cache_tensor_path = os.path.join('./cached',
                                         'ts{}_fs{}_pl{}'.format(self.timestep, self.feature_size, self.pred_len))
        if not os.path.exists(cache_tensor_path):
            os.makedirs(cache_tensor_path)

        self.x_test = csv_to_torch(cache_tensor_path + '/x_test.pt')
        self.y_test = csv_to_torch(cache_tensor_path + '/y_test.pt')

        print("x_test shape: ", self.x_test.shape)
        print("y_test shape: ", self.y_test.shape)

        self.data_x = self.x_test
        self.data_y = self.y_test

    def __getitem__(self, index):
        raw = self.queueIds_df.iloc[index]
        start = raw['test_start']
        end = raw['test_end']
        queueId = raw['QUEUE_ID']
        queue_ids = torch.full((self.timestep, 1), queueId, dtype=torch.long)
        if queueId == 85153:
            # end = end - 200
            end = end - 400
            return self.data_x[end], self.data_y[end], queue_ids
        return self.data_x[end], self.data_y[end], queue_ids

    def __len__(self):
        return len(self.queueIds)

    def inverse_transform(self, data):
        if torch.is_tensor(data):
            data = data.cpu().detach().numpy()
        return self.dataset_obj.inverse_transform(data)

    def inverse_transform_y(self, data):
        if torch.is_tensor(data):
            # data = data.cpu().detach().numpy().reshape(-1, 1)
            data = data.cpu().detach().numpy()
        return self.dataset_obj.inverse_transform_y(data)


class Dataset_Lastest(Dataset):
    def __init__(self, args, data_path, size, flag='train',
                 features='S',
                 target='CPU_USAGE', ratio=0.8, scale=True, scale_type='standard', inverse=False, cols=None,
                 embed='timeF',
                 freq='min'):
        # size [timestep, pred_len]
        # info
        self.timestep = size[0]
        self.feature_size = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'all']
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

        timeenc = 0 if embed != 'timeF' else 1
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()

    def __read_data__(self):
        # 归一化器
        if self.scale_type == 'standard':
            self.scaler = StandardScaler()
            self.scaler_y = StandardScaler()
        elif self.scale_type == 'minmax':
            self.scaler = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        # 读取数据并预处理
        # 默认第一列时间戳为index
        data_df = pd.read_csv(self.data_path)
        # print("读取到本地csv数据： \n", data_df.head())
        print("加载{}数据集...".format(self.flag))

        # 特征工程
        # Convert timestamp to readable datetime
        data_df['datetime'] = pd.to_datetime(data_df['timestamp'], unit='ms')
        # Extract time features
        data_df['hour'] = data_df['datetime'].dt.hour
        data_df['day_of_week'] = data_df['datetime'].dt.dayofweek
        data_df['day'] = data_df['datetime'].dt.day
        data_df['weekday'] = data_df['datetime'].dt.weekday
        data_df['month'] = data_df['datetime'].dt.month

        # Encode QUEUE_ID as category codes
        # data_df['QUEUE_ID'] = data_df['QUEUE_ID'].astype('category').cat.codes

        data_df['CU'] = data_df['CU'].astype('category').cat.codes
        data_df['QUEUE_TYPE'] = data_df['QUEUE_TYPE'].astype('category').cat.codes

        '''
        data_df.columns: ['timestamp(index)', target feature, ...(other features)]
        '''
        # cols = list(data_df.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(data_df.columns)
            cols.remove(self.target)
            cols.remove('QUEUE_ID')
            cols.remove('datetime')
            cols.remove('timestamp')
        data_df = data_df[[self.target] + cols + ['QUEUE_ID']]

        if self.features == 'S':
            # 单元预测，只需要一个元
            df_data = data_df[[self.target]]
        else:
            df_data = data_df

        print("columns列数为：", len(data_df.columns))

        cache_tensor_path = os.path.join('./cached',
                                         'ts{}_fs{}_pl{}'.format(self.timestep, self.feature_size, self.pred_len))
        if not os.path.exists(cache_tensor_path):
            os.makedirs(cache_tensor_path)

        # 检查本地结果文件是否存在，如果存在直接返回
        x_train_cache_tensor_path = cache_tensor_path + '/x_train_{}.pt'.format(self.flag)
        y_train_cache_tensor_path = cache_tensor_path + '/y_train_{}.pt'.format(self.flag)
        x_test_cache_tensor_path = cache_tensor_path + '/x_test.pt'
        y_test_cache_tensor_path = cache_tensor_path + '/y_test.pt'
        queueIds_train_cache_tensor_path = cache_tensor_path + '/queueIds_train.pt'
        queueIds_test_cache_tensor_path = cache_tensor_path + '/queueIds_test.pt'
        x_train_cache_reshaped_path = cache_tensor_path + '/x_train_reshaped.npy'
        y_train_cache_reshaped_path = cache_tensor_path + '/y_train_reshaped.npy'
        if os.path.exists(x_train_cache_reshaped_path) and os.path.exists(y_train_cache_reshaped_path):
            x_train_cache_reshaped = np.load(x_train_cache_reshaped_path)
            y_train_cache_reshaped = np.load(y_train_cache_reshaped_path)
            self.scaler.fit(x_train_cache_reshaped)
            self.scaler_y.fit(y_train_cache_reshaped)
        if self.flag == 'train' or self.flag == 'all':
            if os.path.exists(x_train_cache_tensor_path) and os.path.exists(y_train_cache_tensor_path):
                self.data_x = csv_to_torch(x_train_cache_tensor_path)
                self.data_y = csv_to_torch(y_train_cache_tensor_path)
                self.queueIds = csv_to_torch(queueIds_train_cache_tensor_path)
                print("读取本地训练集缓存数据： \n", self.data_x.shape, self.data_y.shape)
                return
        else:
            if os.path.exists(x_test_cache_tensor_path) and os.path.exists(y_test_cache_tensor_path):
                self.data_x = csv_to_torch(x_test_cache_tensor_path)
                self.data_y = csv_to_torch(y_test_cache_tensor_path)
                self.queueIds = csv_to_torch(queueIds_test_cache_tensor_path)
                print("读取本地测试集缓存数据： \n", self.data_x.shape, self.data_y.shape)
                return

        X_train_all, X_test_all, y_train_all, y_test_all, queueIds_train_all, queueIds_test_all = self.prepare_data(
            df_data)

        print("X_train_all shape: ", X_train_all.shape)
        print("X_test_all shape: ", X_test_all.shape)
        print("y_train_all shape: ", y_train_all.shape)
        print("y_test_all shape: ", y_test_all.shape)
        print("queueIds_train_all shape: ", queueIds_train_all.shape)
        print("queueIds_test_all shape: ", queueIds_test_all.shape)

        # 重塑数据为二维数组进行归一化
        train_nums, num_timesteps, train_features = X_train_all.shape
        test_nums, num_timesteps, test_features = X_test_all.shape

        X_train_all_reshaped = X_train_all.reshape(-1, train_features).astype(np.float32)
        X_test_all_reshaped = X_test_all.reshape(-1, test_features).astype(np.float32)
        y_train_all_reshaped = y_train_all.reshape(-1, 1).astype(np.float32)
        y_test_all_reshaped = y_test_all.reshape(-1, 1).astype(np.float32)

        np.save(x_train_cache_reshaped_path, X_train_all_reshaped)
        np.save(y_train_cache_reshaped_path, y_train_all_reshaped)
        self.scaler.fit(X_train_all_reshaped)
        self.scaler_y.fit(y_train_all_reshaped)
        X_train_all_scaled = self.scaler.transform(X_train_all_reshaped)
        X_test_all_scaled = self.scaler.transform(X_test_all_reshaped)
        y_train_all_scaled = self.scaler_y.transform(y_train_all_reshaped)
        y_test_all_scaled = self.scaler_y.transform(y_test_all_reshaped)

        # 将归一化后的数据重塑回原始三维形状
        X_train_all = X_train_all_scaled.reshape(train_nums, num_timesteps, train_features)
        X_test_all = X_test_all_scaled.reshape(test_nums, num_timesteps, test_features)
        y_train_all = y_train_all_scaled.reshape(train_nums, self.args.pre_len, 1)
        y_test_all = y_test_all_scaled.reshape(test_nums, self.args.pre_len, 1)

        x_train_tensor: torch.Tensor = torch.from_numpy(X_train_all).to(torch.float32)
        y_train_tensor: torch.Tensor = torch.from_numpy(y_train_all).to(torch.float32)
        x_test_tensor: torch.Tensor = torch.from_numpy(X_test_all).to(torch.float32)
        y_test_tensor: torch.Tensor = torch.from_numpy(y_test_all).to(torch.float32)
        queueIds_train_tensor: torch.Tensor = torch.from_numpy(queueIds_train_all).to(torch.float32)
        queueIds_test_tensor: torch.Tensor = torch.from_numpy(queueIds_test_all).to(torch.float32)

        print("x_train shape: ", x_train_tensor.shape)
        print("y_train shape: ", y_train_tensor.shape)
        print("x_test shape: ", x_test_tensor.shape)
        print("y_test shape: ", y_test_tensor.shape)
        print("queueIds_train shape: ", queueIds_train_tensor.shape)
        print("queueIds_test shape: ", queueIds_test_tensor.shape)

        if self.flag == 'train' or self.flag == 'all':
            self.data_x = x_train_tensor
            self.data_y = y_train_tensor
            self.queueIds = queueIds_train_tensor
        else:
            self.data_x = x_test_tensor
            self.data_y = y_test_tensor
            self.queueIds = queueIds_test_tensor

        # 将数据保存到本地
        torch_to_csv(x_train_tensor, x_train_cache_tensor_path)
        torch_to_csv(y_train_tensor, y_train_cache_tensor_path)
        torch_to_csv(x_test_tensor, x_test_cache_tensor_path)
        torch_to_csv(y_test_tensor, y_test_cache_tensor_path)
        torch_to_csv(queueIds_train_tensor, queueIds_train_cache_tensor_path)
        torch_to_csv(queueIds_test_tensor, queueIds_test_cache_tensor_path)

    # 创建时间窗口并分割数据，包含归一化
    def prepare_data(self, data):
        # 分割数据前，按Queue_ID分组
        group_data = data.groupby('QUEUE_ID')

        # 存储分割后的数据
        X_train_all, X_test_all, y_train_all, y_test_all, queueId_train_all, queueId_test_all = [], [], [], [], [], []

        for queue_id, group in group_data:
            x_train, y_train, x_test, y_test, dataX, dataY, queueIds_train, queueIds_test = self.__split_data__(group,
                                                                                                                self.timestep,
                                                                                                                self.feature_size,
                                                                                                                self.pred_len)

            # 存储数据
            X_train_all.append(x_train)
            X_test_all.append(x_test)
            y_train_all.append(y_train)
            y_test_all.append(y_test)
            queueId_train_all.append(queueIds_train)
            queueId_test_all.append(queueIds_test)

        # 合并所有分割后的数据
        X_train_all = np.concatenate(X_train_all, axis=0)
        X_test_all = np.concatenate(X_test_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)
        y_test_all = np.concatenate(y_test_all, axis=0)
        queueId_train_all = np.concatenate(queueId_train_all, axis=0)
        queueId_test_all = np.concatenate(queueId_test_all, axis=0)

        return X_train_all, X_test_all, y_train_all, y_test_all, queueId_train_all, queueId_test_all

    def __split_data__(self, data, timestep: int, feature_size: int,
                       pred_len: int):
        """
        形成训练数据，例如12345789 12-3456789
        :param data: 数据
        :param timestep: 历史时间步的长度
        :param feature_size: 特征数
        :return:
        """
        dataX = []  # 保存X
        dataY = []  # 保存Y
        queueIds = []
        # print(data.shape, timestep, feature_size, pred_len)

        # 将整个窗口的数据保存到X中，将未来一天保存到Y中
        for index in range(len(data) - timestep - pred_len + 1):
            # 第一列是Target, CPU_USAGE
            dataX.append(data[index: index + timestep])
            dataY.append(data.iloc[index + timestep: index + timestep + pred_len, 0].tolist())
            queueIds.append(data.iloc[index: index + timestep, -1].tolist())

        dataX = np.array(dataX)
        dataY = np.array(dataY)
        queueIds = np.array(queueIds)

        # 获取训练集大小
        train_size = int(np.round(self.ratio * dataX.shape[0]))

        # 划分训练集、测试集
        x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
        y_train = dataY[: train_size].reshape(-1, pred_len, 1)
        queueIds_train = queueIds[: train_size].reshape(-1, timestep, 1)

        x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
        y_test = dataY[train_size:].reshape(-1, pred_len, 1)
        queueIds_test = queueIds[train_size:].reshape(-1, timestep, 1)

        return x_train, y_train, x_test, y_test, dataX, dataY.reshape(-1, pred_len, 1), queueIds_train, queueIds_test

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], self.queueIds[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def inverse_transform_y(self, data):
        if torch.is_tensor(data):
            data = data.cpu().detach().numpy()
        return self.scaler_y.inverse_transform(data)
