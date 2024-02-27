import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from models.RevIN.RevIN import RevIN
from models.seqformer.DSFormer import MLP


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
    def __init__(self, feature_size, hidden_size, num_layers, dec_layers, num_heads, dropout, device, pre_len, timestep,
                 output_size,
                 use_RevIN=False, forward_expansion=8, dec_type='decoder'):
        super(Transformer, self).__init__()
        self.pre_len = pre_len
        self.use_RevIN = use_RevIN
        self.dec_type = dec_type
        self.input_fc = nn.Linear(feature_size, hidden_size)
        self.pos_emb = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=forward_expansion * hidden_size,
            batch_first=True,
            dropout=dropout,
            device=device
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=forward_expansion * hidden_size,
            batch_first=True,
            device=device
        )
        self.output = nn.Embedding(pre_len, hidden_size)
        self.output_pos = nn.Embedding(pre_len, hidden_size)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        assert dec_type in ['mlp', 'decoder']
        # 投影层，可替代解码器
        if dec_type == 'mlp':
            self.mlp = MLP(timestep, pre_len, hidden_size, 2, dropout, activation='gelu')
        elif dec_type == 'decoder':
            self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        else:
            raise Exception('不支持其他类型的解码器')

        self.fc = nn.Linear(hidden_size, output_size)

        if use_RevIN:
            self.revin = RevIN(feature_size)

        print("Number Parameters: transformer", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, queue_ids):
        batch_size = x.shape[0]
        if self.use_RevIN:
            x = self.revin(x, 'norm')
        # print(x.size())  # [256, 126, 8]
        x = self.input_fc(x)  # [256, 126, 256]
        x = self.pos_emb(x)  # [256, 126, 256]
        x = self.encoder(x)  # [256, 126, 256]

        # output_size, batch_size, hidden_size
        output = self.output.weight.unsqueeze(1).repeat(1, batch_size, 1)
        output_pos = self.output_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)
        output = output.permute(1, 0, 2)
        output_pos = output_pos.permute(1, 0, 2)

        if self.dec_type == 'mlp':
            x = x.permute(0, 2, 1)
            output = self.mlp(x).permute(0, 2, 1)
        elif self.dec_type == 'decoder':
            output = self.decoder(output, x)
        else:
            raise Exception('不支持其他类型的解码器')
        output = self.fc(output)

        if self.use_RevIN:
            output = self.revin(output, 'denorm')

        return output[:, -self.pre_len:, 0:1]
