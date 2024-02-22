import numpy as np
import torch
from torch import nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import AttentionLayer, ProbAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer, ConvLayer
from models.RevIN.RevIN import RevIN
from models.seqformer.seqformer import series_decomp


class DsFormer(nn.Module):
    def __init__(self, timestep, feature_size, hidden_size, enc_layers, num_heads, ffn_hidden_size, dropout, pred_len,
                 use_RevIN=False, moving_avg=25, w_lin=1.0, factor=1, output_attention=False,
                 activation='gelu', conv=True):
        super(DsFormer, self).__init__()

        self.use_RevIN = use_RevIN
        self.pre_len = pred_len

        self.decompsition = series_decomp(moving_avg)
        self.Linear_Trend = nn.Linear(timestep, pred_len)
        self.Linear_Trend.weight = nn.Parameter(
            (1 / timestep) * torch.ones([pred_len, timestep]))

        self.enc_embedding = DataEmbedding(feature_size, hidden_size, 'timeF', 't',
                                           dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), hidden_size, num_heads),
                    hidden_size,
                    ffn_hidden_size,
                    dropout=dropout,
                    activation=activation
                ) for l in range(enc_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size)
        )

        self.w_dec = torch.nn.Parameter(torch.FloatTensor([w_lin] * feature_size), requires_grad=True)

        # 投影层，可替代解码器
        self.projection = nn.Linear(hidden_size, pred_len, bias=True)

        if use_RevIN:
            self.revin = RevIN(feature_size)

        print("Number Parameters: dsformer", self.get_n_params())

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
        trend_output = self.Linear_Trend(trend_init).permute(0, 2, 1)  # trend_output: [B, P, D]

        # 对于非线性部分和周期性特征进行学习
        x = seasonal_init

        # Embedding
        enc_out = self.enc_embedding(x, x_mark=None)

        # timeStep, batch_size, hidden_size
        enc_out, attns = self.encoder(enc_out)

        dec_out = self.projection(enc_out)[:, :, :feature_size]

        # 将季节性与趋势性相加
        output = dec_out + self.w_dec * trend_output  # output: [B, P, D]

        if self.use_RevIN:
            output = self.revin(output, 'denorm')

        return output[:, -self.pre_len:, 0:1]
