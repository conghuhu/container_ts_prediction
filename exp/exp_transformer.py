import gc
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_loader import Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.Transformer.transformer import Transformer
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics import metric, write_to_file
from utils.plot_tools import plot_loss_data, closePlots
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_Transformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Transformer, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = Transformer(args)

        return model

    def _load_data(self):
        args = self.args
        train_data_set = Dataset_Custom(
            args=args,
            data_path=args.data_path,
            flag="train",
            size=[args.timestep, args.feature_size, args.pre_len],
            features=args.features,
            target=args.target,
            scale_type=args.scale_type,
            inverse=args.inverse,
        )
        test_data_set = Dataset_Custom(
            args=args,
            data_path=args.data_path,
            flag="test",
            size=[args.timestep, args.feature_size, args.pre_len],
            features=args.features,
            target=args.target,
            scale_type=args.scale_type,
            inverse=args.inverse,
        )
        pred_data_set = Dataset_Pred(
            args=args,
            data_path=args.data_path,
            dataset_obj=train_data_set,
            flag="pred",
            size=[args.timestep, args.feature_size, args.pre_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
        )
        self.train_data_set = train_data_set
        self.test_data_set = test_data_set
        self.pred_data_set = pred_data_set

        self.train_loader = DataLoader(self.train_data_set,
                                       self.args.batch_size,
                                       shuffle=False)
        self.test_loader = DataLoader(self.test_data_set,
                                      self.args.batch_size,
                                      shuffle=False)
        self.pred_loader = DataLoader(self.pred_data_set,
                                      1,
                                      shuffle=False)

    def _get_data(self, flag):
        args = self.args
        if flag == 'train':
            return self.train_data_set, self.train_loader
        elif flag == 'test':
            return self.test_data_set, self.test_loader
        else:
            return self.pred_data_set, self.pred_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_loss_function(self):
        loss_name = self.args.loss_name
        if loss_name == 'MSE':
            return nn.MSELoss().to(self.device)
        elif loss_name == 'MAPE':
            return mape_loss().to(self.device)
        elif loss_name == 'MASE':
            return mase_loss().to(self.device)
        elif loss_name == 'SMAPE':
            return smape_loss().to(self.device)

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

        results_train_loss = []
        results_test_loss = []

        for epoch in range(self.args.epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
                iter_count += 1

                model_optim.zero_grad()
                if i == 0:
                    print("batch_x shape: ", batch_x.shape)
                    print("batch_y shape: ", batch_y.shape)
                # batch_x shape: [batch_size, timestep, feature_size]
                # batch_y shape: [batch_size, pred_len, 1]
                pred, true = self._process_one_batch(train_data_set, batch_x, batch_y)
                if i == 0:
                    print("pred shape: ", pred.shape)
                    print("true shape: ", true.shape)
                # pred shape: [batch_size, pred_len, 1]
                # true shape: [batch_size, pred_len, 1]
                loss = loss_function(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\niters: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            test_loss = self.vali(test_data_set, test_loader, loss_function)

            results_train_loss.append(train_loss)
            results_test_loss.append(test_loss)

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

        plot_loss_data(results_train_loss, self.args.loss_name, self.args.setting, 'train')
        plot_loss_data(results_test_loss, self.args.loss_name, self.args.setting, 'test')

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, load=False):
        test_data_set, test_loader = self._get_data(flag='test')
        if load:
            print('loading model')
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        # 核心是计算metrics
        preds: list = []
        trues: list = []

        # 仅仅用来展示测试集的整体预测情况
        results = []
        labels = []

        with torch.no_grad():
            for batch_x, batch_y in tqdm(test_loader):
                # pred shape: [batch_size, pred_len, 1]
                pred, true = self._process_one_batch(test_data_set, batch_x, batch_y)
                pred = pred.detach().cpu().numpy()
                true = true.detach().cpu().numpy()
                # 只取每个多步预测的第一个值, 只取第一个值的 pred shape: [batch_size, 1]
                pred_inverse = test_data_set.inverse_transform_y(pred[:, 0, :])
                true_inverse = test_data_set.inverse_transform_y(true[:, 0, :])
                preds.append(pred)
                trues.append(true)

                for i in range(len(pred)):
                    results.append(pred_inverse[i][-1])
                    labels.append(true_inverse[i][-1])

        preds: np.ndarray = np.concatenate(preds, axis=0)
        trues: np.ndarray = np.concatenate(trues, axis=0)
        print('测试集整体预测结果和真实的shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 输出测试集的指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse: {}, mae: {}, rmse: {}, mape: {}, mspe: {}'.format(mse, mae, rmse, mape, mspe))
        write_to_file(setting, mse, mae, rmse, mape, mspe)

        # 将数据保存到本地
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # 画出测试集拟合曲线
        plt.figure(figsize=(15, 8))
        # 绘制历史数据
        plt.plot(labels, label='TrueValue')
        # 绘制预测数据
        plt.plot(results, label='Prediction')

        # 添加标题和图例
        plt.title("test state")
        plt.legend()
        plt.show()

        return

    def predict(self, setting, load=False, args=None):
        if args is None:
            args = self.args

        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        folder_path = './predict_imgs/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        for i, (batch_x, batch_y, queueId) in enumerate(tqdm(pred_loader)):
            queueId = queueId.item()
            history_data: np.ndarray = pred_data.inverse_transform_y(batch_x[:, :, 0].reshape(args.timestep, 1))
            pred, true = self._process_one_batch(pred_data, batch_x, batch_y)
            # pred.shape [batchSize=1, pre_len, 1]
            pred = pred[0]
            pred = pred_data.inverse_transform_y(pred)
            # pred.shape [pre_len, 1]
            true = true[0]
            true = pred_data.inverse_transform_y(true)
            # true.shape [pre_len, 1]

            # 真实展示的数据
            true_show_data = np.concatenate([history_data[:, -1], true[:, -1]], axis=0)
            # true_show_data.shape [timestep+pre_len]

            # 绘图
            plt.figure()
            if args.features == 'MS' or args.features == 'S':
                # print("true_show_data: \n", true_show_data)
                # print("pred data: \n", pred)
                plt.plot(range(len(true_show_data)), true_show_data,
                         label='True Values')
                plt.plot(range(len(true_show_data) - args.pre_len, len(true_show_data)), pred[:, -1],
                         label='Predicted Values')
            else:
                print('未实现多元预测多元的可视化')
                return
            # 添加图例
            plt.legend()
            plt.style.use('ggplot')
            # 添加标题和轴标签
            plt.title('Past vs Predicted Future Values, QUEUE_ID: {}'.format(queueId))
            plt.xlabel('Time Point')
            plt.ylabel(args.target)
            # 在特定索引位置画一条直线
            plt.axvline(len(true_show_data) - args.pre_len, color='blue', linestyle='--', linewidth=2)
            # 显示图表
            plt.savefig(folder_path + '{}_{}_forcast.png'.format(args.target, queueId))
            # 由于存在限流，只展示前25张图片
            if i < 25:
                plt.show()

            closePlots()

    def _process_one_batch(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        y_pred = self.model(batch_x)
        if self.args.inverse:
            y_pred = dataset_object.inverse_transform(y_pred)

        return y_pred, batch_y
