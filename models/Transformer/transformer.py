import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


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
    def __init__(self, feature_size, hidden_size, num_layers, num_heads, dropout, device, pre_len, timestep):
        super(Transformer, self).__init__()
        # embed_dim = head_dim * num_heads?
        self.pre_len = pre_len
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
        self.fc1 = nn.Linear(timestep * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, pre_len)

        print("Number Parameters: cnn-lstm-attention", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x):
        # print(x.size())  # [256, 126, 8]
        x = self.input_fc(x)  # [256, 126, 256]
        x = self.pos_emb(x)  # [256, 126, 256]
        x = self.encoder(x)  # [256, 126, 256]
        # 不经过解码器
        x = x.flatten(start_dim=1)  # [256, 32256]
        x = self.fc1(x)  # [256, 256]
        output = self.fc2(x)  # [256, 256]

        return output.unsqueeze(-1)
