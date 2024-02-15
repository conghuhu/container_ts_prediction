import math
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


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

        # B,L,D
        return shortcut + x.transpose(1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(self.pe[:x.size(0), :, :])


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.0, pre_norm=False):
        super().__init__()

        self.num_layers = num_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )

            self.conv_layers.append(
                ConvolutionModule(
                    channels=d_model,
                    kernel_size=3,
                    stride=1
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )

    def forward(self, output, query_pos):
        for i in range(self.num_layers):
            output_att = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_pos
            )

            output = output.transpose(1, 0)
            output = self.conv_layers[i](output)
            output = output.transpose(1, 0)

            output = self.transformer_ffn_layers[i](
                output_att + output
            )

        return output


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.0, pre_norm=False):
        super().__init__()

        self.num_layers = num_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

    def forward(self, output, src, pos, query_pos):
        for i in range(self.num_layers):
            output = self.transformer_cross_attention_layers[i](
                output, src,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos, query_pos=query_pos
            )

            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_pos
            )

            output = self.transformer_ffn_layers[i](
                output
            )

        return output


class SeqFormer(nn.Module):
    def __init__(self, timestep, feature_size, hidden_size, num_layers, num_heads, ffn_hidden_size, dropout, pre_norm,
                 output_size, pre_len):
        super(SeqFormer, self).__init__()

        self.fc_all = nn.Linear(feature_size, hidden_size)
        self.fc_cpu = nn.Linear(1, hidden_size)
        self.fc_fuse = nn.Linear(hidden_size, hidden_size)
        self.fc_input = nn.Linear(feature_size, hidden_size)

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_size,
            dropout=dropout,
            pre_norm=pre_norm
        )
        self.x_pos = PositionalEncoding(hidden_size, max_len=timestep)

        self.output = nn.Embedding(pre_len, hidden_size)
        self.output_pos = nn.Embedding(pre_len, hidden_size)

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_size,
            dropout=dropout,
            pre_norm=pre_norm
        )

        self.decoder_norm = nn.LayerNorm(hidden_size)

        self.fc_output = nn.Linear(hidden_size, output_size)

        print("Number Parameters: seqformer", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, queue_ids):
        # x.shape(batch_size, timeStep, feature_size)
        batch_size = x.shape[0]

        # timeStep, batch_size, feature_size
        x = x.transpose(1, 0)

        # timeStep, batch_size, hidden_size
        x = self.fc_fuse(self.fc_cpu(x[:, :, 0].unsqueeze(-1)) + self.fc_all(x))
        # x = self.fc_input(x)
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

        return output
