import gc
import os
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_loader import Dataset_Custom
from exp.exp_basic import Exp_Basic
from models.CNN_LSTM.CNN_LSTM_net import CNN_LSTM_Attention
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_CNN_LSTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_CNN_LSTM, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = CNN_LSTM_Attention(args.feature_size, args.timestep, args.hidden_size, args.num_layers,
                                   args.out_channels, args.num_heads, args.output_size)
        return model

    def _load_data(self):
        args = self.args
        train_data_set = Dataset_Custom(
            args=args,
            data_path=args.data_path,
            flag="train",
            size=[args.timestep, args.feature_size, args.output_size],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
        )
        test_data_set = Dataset_Custom(
            args=args,
            data_path=args.data_path,
            flag="test",
            size=[args.timestep, args.feature_size, args.output_size],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
        )
        self.train_data_set = train_data_set
        self.test_data_set = test_data_set

        self.train_loader = DataLoader(self.train_data_set,
                                       self.args.batch_size,
                                       shuffle=False)
        self.test_loader = DataLoader(self.test_data_set,
                                      self.args.batch_size,
                                      shuffle=False)

    def _get_data(self, flag):
        args = self.args
        if flag == 'train':
            return self.train_data_set, self.train_loader
        else:
            return self.test_data_set, self.test_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_loss_function(self):
        loss_function = nn.MSELoss()
        loss_function = loss_function.to(self.device)
        return loss_function

    def vali(self, vali_data_set, vali_loader, loss_function):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data_set, batch_x, batch_y)
                loss = loss_function(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data_set, train_loader = self._get_data(flag='train')
        test_data_set, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        loss_function = self._select_loss_function()

        for epoch in range(self.args.epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
                iter_count += 1

                model_optim.zero_grad()
                # batch_x shape: [batch_size, timestep, feature_size]
                # batch_y shape: [batch_size, pred_len]
                pred, true = self._process_one_batch(train_data_set, batch_x, batch_y)
                # pred shape: [batch_size, pred_len]
                # true shape: [batch_size, pred_len]
                loss = loss_function(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\niters: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            test_loss = self.vali(test_data_set, test_loader, loss_function)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss))
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # GC优化
            gc.collect()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data_set, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds: list = []
        trues: list = []

        for batch_x, batch_y in tqdm(test_loader):
            pred, true = self._process_one_batch(test_data_set, batch_x, batch_y)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds: np.ndarray = np.concatenate(preds, axis=0)
        trues: np.ndarray = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse: {}, mae: {}, rmse: {}, mape: {}, mspe: {}'.format(mse, mae, rmse, mape, mspe))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # 增加一个绘图功能

        return

    def calculate_mse(self, y_true, y_pred):
        # 均方误差
        mse = np.mean(np.abs(y_true - y_pred))
        return mse

    def predict(self, setting, load=False, args=None):
        df = pd.read_csv(args.data_path)
        # 历史数据，仅供展示
        history_data = df[args.target].iloc[-(args.timestep + args.output_size):].reset_index(drop=True)
        pre_data = pd.read_csv(args.data_path)
        pre_data['date'] = pd.to_datetime(pre_data['date'])
        columns = ['forecast' + column for column in pre_data.columns]

        pre_data.reset_index(inplace=True, drop=True)
        # 预测的长度
        pre_length = args.output_size
        # 数据都读取进来
        dict_of_lists = {column: [] for column in columns}

        results = []
        for i in range(int(len(pre_data) / pre_length)):
            if i == 0:
                pred_data, pred_loader = self._get_data(flag='pred')
            else:
                pred_data, pred_loader = self._get_data(flag='pred', pre_data=pre_data.iloc[:i * pre_length])

            print(f'预测第{i + 1} 次')
            if load:
                path = os.path.join(self.args.checkpoints, setting)
                best_model_path = path + '/' + 'checkpoint.pth'
                self.model.load_state_dict(torch.load(best_model_path))

            self.model.eval()

            for ii, (batch_x, batch_y) in enumerate(tqdm(pred_loader)):
                pred, true = self._process_one_batch(
                    pred_data, batch_x, batch_y)
                print("pred shape: ", pred.shape)
                # pred shape: 1 * pred_len * 1
                pred = pred_data.inverse_transform(pred)
                if args.features == 'MS' or args.features == 'S':
                    for iii in range(args.pred_len):
                        results.append(pred[0][iii][0].detach().cpu().numpy())
                else:
                    for j in range(args.enc_in):
                        for iii in range(args.pred_len):
                            dict_of_lists[columns[j]].append(pred[0][iii][j].detach().cpu().numpy())
                print(pred)
            if not args.is_rolling_predict:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>不进行滚动预测<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                break

        # 将预测结果导出到本地
        if not args.is_rolling_predict:
            if args.features == 'MS' or args.features == 'S':
                df = pd.DataFrame({'forecast{}'.format(args.target): pre_data[args.target]})
                df.to_csv('Interval-{}'.format(args.data_path), index=False)
            else:
                df = pd.DataFrame(dict_of_lists)
                df.to_csv('Interval-{}'.format(args.data_path), index=False)
        else:
            if args.features == 'MS' or args.features == 'S':
                df = pd.DataFrame({'date': pre_data['date'], '{}'.format(args.target): pre_data[args.target],
                                   'forecast{}'.format(args.target): pre_data[args.target]})
                df.to_csv('Interval-{}'.format(args.data_path), index=False)
            else:
                df = pd.DataFrame(dict_of_lists)
                new_df = pd.concat((pre_data, df), axis=1)
                new_df.to_csv('Interval-{}'.format(args.data_path), index=False)

        # 绘图
        plt.figure()
        if args.is_rolling_predict:
            pre_len = len(dict_of_lists['forecast' + args.target])
            if args.features == 'MS' or args.features == 'S':
                # print(results)
                # print("===============")
                print(range(len(history_data), len(history_data) + pre_len))
                plt.plot(range(len(history_data)), history_data,
                         label='Past Actual Values')
                plt.plot(range(len(history_data), len(history_data) + pre_len),
                         pre_data[args.target][:pre_len].tolist(), label='Predicted Actual Values')
                plt.plot(range(len(history_data), len(history_data) + pre_len), results,
                         label='Predicted Future Values')
            else:
                plt.plot(range(len(history_data)), history_data,
                         label='Past Actual Values')
                plt.plot(range(len(history_data), len(history_data) + pre_len),
                         pre_data[args.target][:pre_len].tolist(), label='Predicted Actual Values')
                plt.plot(range(len(history_data), len(history_data) + pre_len), dict_of_lists['forecast' + args.target],
                         label='Predicted Future Values')
        else:
            if args.features == 'MS' or args.features == 'S':
                plt.plot(range(len(history_data)), history_data,
                         label='Past Actual Values')
                plt.plot(range(len(history_data) - len(results), len(history_data)), results,
                         label='Predicted Future Values')
            else:
                plt.plot(range(len(history_data)), history_data,
                         label='Past Actual Values')
                plt.plot(range(len(history_data), len(history_data) + len(dict_of_lists['forecast' + args.target])),
                         dict_of_lists['forecast' + args.target], label='Predicted Future Values')
        # 添加图例
        plt.legend()
        plt.style.use('ggplot')
        # 添加标题和轴标签
        plt.title('Past vs Predicted Future Values')
        plt.xlabel('Time Point')
        plt.ylabel('Value')
        # 在特定索引位置画一条直线
        plt.axvline(x=len(history_data) - len(results), color='blue', linestyle='--', linewidth=2)
        # 显示图表
        plt.savefig('forcast.png')
        plt.show()
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        batch_y = batch_y.reshape(-1, 1)

        return outputs, batch_y
