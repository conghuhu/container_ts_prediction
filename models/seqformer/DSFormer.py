import math
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from layers.Embed import DataEmbedding
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

class SeqFormer(nn.Module):
    def __init__(self, timestep, feature_size, hidden_size, enc_layers, dec_layers, num_heads, ffn_hidden_size, dropout,
                 pre_norm,
                 output_size, pred_len, use_RevIN=False, moving_avg=25, w_lin=1.0):
        super(SeqFormer, self).__init__()

        self.use_RevIN = use_RevIN
        self.pre_len = pred_len

        self.decompsition = series_decomp(moving_avg)
        self.Linear_Trend = nn.Linear(timestep, pred_len)
        # self.Linear_Trend.weight = nn.Parameter(
        #     (1 / pred_len) * torch.ones([pred_len, timestep]),
        #     requires_grad=True)
        self.Linear_Trend.weight = nn.Parameter(
            (1 / timestep) * torch.ones([pred_len, timestep]))

        self.enc_embedding = DataEmbedding(feature_size, hidden_size, 'timeF', 't',
                                           dropout)

        self.fc_all = nn.Linear(feature_size, hidden_size)
        self.fc_cpu = nn.Linear(1, hidden_size)
        self.fc_fuse = nn.Linear(hidden_size, hidden_size)
        self.fc_input = nn.Linear(feature_size, hidden_size)

        self.encoder = Encoder(
            num_layers=enc_layers,
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_size,
            dropout=dropout,
            pre_norm=pre_norm
        )
        self.x_pos = PositionalEncoding(hidden_size, max_len=timestep)

        self.output = nn.Embedding(pred_len, hidden_size)
        self.output_pos = nn.Embedding(pred_len, hidden_size)

        self.decoder = Decoder(
            num_layers=dec_layers,
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_size,
            dropout=dropout,
            pre_norm=pre_norm
        )

        self.decoder_norm = nn.LayerNorm(hidden_size)

        self.fc_output = nn.Linear(hidden_size, output_size)

        self.w_dec = torch.nn.Parameter(torch.FloatTensor([w_lin] * feature_size), requires_grad=True)

        # 投影层，可替代解码器
        self.projection = nn.Linear(hidden_size, pred_len, bias=True)

        if use_RevIN:
            self.revin = RevIN(feature_size)

        print("Number Parameters: seqformer", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, queue_ids):
        # x.shape(batch_size, timeStep, feature_size)
        batch_size, timeStep, feature_size = x.shape

        if self.use_RevIN:
            x = self.revin(x, 'norm')

        # 季节与时间趋势性分解
        seasonal_init, trend_init = self.decompsition(x)  # seasonal_init: [B, T, D]  trend_init: [B, T, D]
        # 将维度索引2与维度索引1交换
        trend_init = trend_init.permute(0, 2, 1)  # seasonal_init: [B, D, T]  trend_init: [B, D, T]
        trend_output = self.Linear_Trend(trend_init)  # trend_output: [B, D, P]
        trend_output = trend_output.permute(0, 2, 1)  # trend_output: [B, P, D]

        x = seasonal_init

        # timeStep, batch_size, feature_size
        x = x.transpose(1, 0)

        # timeStep, batch_size, hidden_size
        # x = self.fc_fuse(self.fc_cpu(x[:, :, 0].unsqueeze(-1)) + self.fc_all(x))
        # 消融实验一
        x = self.fc_input(x)
        x_pos = self.x_pos(x)

        # timeStep, batch_size, hidden_size
        x = self.encoder(x, x_pos)

        # output_size, batch_size, hidden_size
        output = self.output.weight.unsqueeze(1).repeat(1, batch_size, 1)
        output_pos = self.output_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # pre_len, batch_size, hidden_size
        output = self.decoder(output, x, x_pos, output_pos)

        # pre_len, batch_size, hidden_size
        output = self.decoder_norm(output)

        # pre_len, batch_size, output_size
        output = self.fc_output(output)

        # batch_size, pre_len, output_size
        output = output.transpose(1, 0)

        # 将季节性与趋势性相加
        output = output + self.w_dec * trend_output  # output: [B, P, D]

        if self.use_RevIN:
            output = self.revin(output, 'denorm')

        return output[:, -self.pre_len:, 0:1]
