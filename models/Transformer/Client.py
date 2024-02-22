import torch
from torch import nn

from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer
from models.RevIN.RevIN import RevIN


class Client(nn.Module):

    def __init__(self, seq_len, feature_size, pred_len, e_layers, n_heads, factor, dropout, d_ff,
                 output_attention=False, activation='gelu', w_lin=1.0, use_RevIN=False):
        super(Client, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.d_model = seq_len
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), self.d_model, n_heads),
                    self.d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.proj = nn.Linear(self.d_model, self.pred_len, bias=True)
        self.Linear = nn.Sequential()
        self.Linear.add_module('Linear', nn.Linear(seq_len, self.pred_len))
        self.w_dec = torch.nn.Parameter(torch.FloatTensor([w_lin] * feature_size), requires_grad=True)
        self.use_RevIN = use_RevIN
        if use_RevIN:
            self.revin_layer = RevIN(feature_size)

    def forward(self, x_enc, queue_ids):
        if self.use_RevIN:
            x_enc = self.revin_layer(x_enc, 'norm')

        # 送入编码器
        enc_out = x_enc.permute(0, 2, 1)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.proj(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        # 趋势分解
        linear_out = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = dec_out[:, -self.pred_len:, :] + self.w_dec * linear_out
        if self.use_RevIN:
            dec_out = self.revin_layer(dec_out, 'denorm')

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, 0:1]
