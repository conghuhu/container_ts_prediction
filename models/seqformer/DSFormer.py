import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import AttentionLayer, ProbAttention
from layers.Transformer_EncDec import Encoder
from models.RevIN.RevIN import RevIN
from models.seqformer.seqformer import series_decomp


class MLP(nn.Module):
    """
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    """

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=128,
                 hidden_layers=2,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class ConvolutionModule(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "layer_norm",
                 bias: bool = True,
                 stride: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.bias = bias
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.norm = nn.LayerNorm(channels) if norm == "layer_norm" else nn.BatchNorm1d(channels)
        self.stride = stride

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias
        )

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        pw_max = self.channels ** -0.5
        dw_max = self.kernel_size ** -0.5
        torch.nn.init.uniform_(self.pointwise_conv1.weight.data, -pw_max, pw_max)
        torch.nn.init.uniform_(self.depthwise_conv.weight.data, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pointwise_conv2.weight.data, -pw_max, pw_max)
        if self.bias:
            torch.nn.init.uniform_(self.pointwise_conv1.bias.data, -pw_max, pw_max)
            torch.nn.init.uniform_(self.depthwise_conv.bias.data, -dw_max, dw_max)
            torch.nn.init.uniform_(self.pointwise_conv2.bias.data, -pw_max, pw_max)

    def forward(self, x: torch.Tensor):
        shortcut = x

        # B,L,D -> B,D,L  D is channel
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        x = self.depthwise_conv(x)

        # layer norm
        x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        x = x.transpose(1, 2)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # 残差连接
        # B,L,D
        return shortcut + x.transpose(1, 2)


class ConvEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", use_conv=False):
        super(ConvEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.conv = ConvolutionModule(
            channels=d_model,
            kernel_size=3,
            stride=1
        ) if use_conv else None

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 自注意力模块
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        att_output = self.norm1(x + self.dropout(new_x))

        # 卷积模块
        if self.conv is not None:
            conv_x = self.conv(x)
            conv_output = self.norm2(conv_x + x)

        y = output = (att_output + conv_output) if self.conv is not None else att_output

        # FFN
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(output + y), attn


class DsFormer(nn.Module):
    def __init__(self, timestep, feature_size, hidden_size, enc_layers, num_heads, ffn_hidden_size, dropout, pred_len,
                 use_RevIN=False, moving_avg=25, w_lin=1.0, factor=1, output_attention=False,
                 activation='gelu', conv=True, dec_type='mlp'):
        super(DsFormer, self).__init__()

        self.use_RevIN = use_RevIN
        self.pre_len = pred_len
        self.dec_type = dec_type
        self.hidden_size = hidden_size

        self.decompsition = series_decomp(moving_avg)
        self.Linear_Trend = nn.Linear(timestep, pred_len)
        self.Linear_Trend.weight = nn.Parameter(
            (1 / timestep) * torch.ones([pred_len, timestep]))

        self.enc_embedding = DataEmbedding(feature_size, hidden_size, 'timeF', 't',
                                           dropout)

        self.encoder = Encoder(
            [
                ConvEncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), hidden_size, num_heads),
                    hidden_size,
                    ffn_hidden_size,
                    dropout=dropout,
                    activation=activation,
                    use_conv=conv
                ) for l in range(enc_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size)
        )

        self.w_dec = torch.nn.Parameter(torch.FloatTensor([w_lin] * feature_size), requires_grad=True)

        assert dec_type in ['mlp', 'linear']
        # 投影层，可替代解码器
        if dec_type == 'mlp':
            self.mlp = MLP(timestep, pred_len, hidden_size, 2, dropout, activation=activation)
        elif dec_type == 'linear':
            self.projection = nn.Linear(timestep, pred_len, bias=True)
        else:
            raise Exception('不支持其他类型的解码器')
        self.fc = nn.Linear(hidden_size, feature_size)

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
        enc_out = self.enc_embedding(x, x_mark=None)  # enc_out: [B, T, hidden_size]

        # timeStep, batch_size, hidden_size
        enc_out, attns = self.encoder(enc_out)  # enc_out: [B, T, hidden_size]

        if self.dec_type == 'mlp':
            enc_out = enc_out.permute(0, 2, 1) # enc_out: [B, D, T]
            dec_out = self.mlp(enc_out).permute(0, 2, 1)  # enc_out: [B, P, D]
        elif self.dec_type == 'linear':
            enc_out = enc_out.permute(0, 2, 1) # enc_out: [B, D, T]
            dec_out = self.projection(enc_out).permute(0, 2, 1) # enc_out: [B, D, P] -> [B, P, D]
        else:
            raise Exception('不支持其他类型的解码器')
        dec_out = self.fc(dec_out) # enc_out: [B, P, F]

        # 将季节性与趋势性相加
        output = dec_out + self.w_dec * trend_output  # output: [B, P, F]

        if self.use_RevIN:
            output = self.revin(output, 'denorm')

        return output[:, -self.pre_len:, 0:1]
