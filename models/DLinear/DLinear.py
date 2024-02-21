import numpy as np
import torch
from torch import nn

from models.RevIN.RevIN import RevIN


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # 滑动平均
        moving_mean = self.moving_avg(x)
        # 季节趋势性
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, timestep, feature_size, pred_len, moving_avg, enc_inc, individual=False, use_RevIN=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(DLinear, self).__init__()
        self.seq_len = timestep
        self.pred_len = pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = enc_inc

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        self.use_RevIN = use_RevIN
        if use_RevIN:
            self.revin = RevIN(feature_size)
        print("Number Parameters: seqformer", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def encoder(self, x):
        # 季节与时间趋势性分解
        seasonal_init, trend_init = self.decompsition(x)  # seasonal_init: [B, T, D]  trend_init: [B, T, D]
        # 将维度索引2与维度索引1交换
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)  # seasonal_init: [B, D, T]  trend_init: [B, D, T]
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                # 使用全连接层得到季节性
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                # 使用全连接层得到趋势性
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
                # 两者共享所有权重
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)  # seasonal_output: [B, D, P]
            trend_output = self.Linear_Trend(trend_init)  # trend_output: [B, D, P]
        # 将季节性与趋势性相加
        x = seasonal_output + trend_output  # x: [B, D, P]
        return x.permute(0, 2, 1)

    def forward(self, x_enc, queue_ids):
        # x_enc 输入shape: [B, T, D]
        if self.use_RevIN:
            x = self.revin(x_enc, 'norm')
        dec_out = self.encoder(x_enc)  # dec_out: [B, P, D]
        if self.use_RevIN:
            dec_out = self.revin(dec_out, 'denorm')
        return dec_out[:, -self.pred_len:, 0:1]  # [B, P, 1]
