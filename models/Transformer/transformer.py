import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from models.RevIN.RevIN import RevIN


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, num_heads, dropout, device, pre_len, timestep,
                 output_size,
                 use_RevIN=False):
        super(Transformer, self).__init__()
        self.pre_len = pre_len
        self.use_RevIN = use_RevIN
        self.input_fc = nn.Linear(feature_size, hidden_size)
        self.pos_emb = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            batch_first=True,
            dropout=dropout,
            device=device
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=4 * hidden_size,
            batch_first=True,
            device=device
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # self.fc = nn.Linear(output_size * hidden_size, output_size)
        self.fc_layers = nn.Sequential(
            nn.Linear(timestep * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pre_len)
        )

        if use_RevIN:
            self.revin = RevIN(feature_size)

        print("Number Parameters: transformer", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, queue_ids):
        # if self.use_RevIN:
        #     x = self.revin(x, 'norm')
        # print(x.size())  # [256, 126, 8]
        x = self.input_fc(x)  # [256, 126, 256]
        x = self.pos_emb(x)  # [256, 126, 256]
        x = self.encoder(x)  # [256, 126, 256]
        # 不经过解码器
        x = x.flatten(start_dim=1)  # [B, T*hidden_size]
        # x = self.fc1(x)  # [256, 256]
        # output = self.fc2(x)  # [256, 256]
        output = self.fc_layers(x)  # 通过Sequential模块处理
        # if self.use_RevIN:
        #     output = self.revin(output, 'denorm')

        return output.unsqueeze(-1)
