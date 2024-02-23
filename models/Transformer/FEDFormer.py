import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AutoCorrelation import AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, Decoder, DecoderLayer
from layers.Embed import DataEmbedding
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletTransform, MultiWaveletCross
from models.seqformer.seqformer import series_decomp


class FEDFormer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, version='Fourier', mode_select='random', modes=32, embed='timeF', freq='t'):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(FEDFormer, self).__init__()
        self.seq_len = configs.timestep
        self.label_len = configs.label_len
        self.pred_len = configs.pre_len

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(configs.feature_size, configs.hidden_size, embed, freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.feature_size, configs.hidden_size, embed, freq,
                                           configs.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.hidden_size, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs.hidden_size, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=configs.hidden_size,
                                                  out_channels=configs.hidden_size,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=configs.hidden_size,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.hidden_size,
                                            out_channels=configs.hidden_size,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.hidden_size,
                                            out_channels=configs.hidden_size,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.hidden_size,
                                                      out_channels=configs.hidden_size,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select,
                                                      num_heads=configs.num_heads)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.hidden_size, configs.num_heads),
                    configs.hidden_size,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.enc_layers)
            ],
            norm_layer=my_Layernorm(configs.hidden_size)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.hidden_size, configs.num_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.hidden_size, configs.num_heads),
                    configs.hidden_size,
                    configs.output_size,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.dec_layers)
            ],
            norm_layer=my_Layernorm(configs.hidden_size),
            projection=nn.Linear(configs.hidden_size, configs.output_size, bias=True)
        )

        print("Number Parameters: FEDformer", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forecast(self, x_enc):
        # [B, T, F]
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)  # mean [B, T, F]
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg [B, T, F] [B, T, F]
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark=None)
        dec_out = self.dec_embedding(seasonal_init, x_mark=None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def forward(self, x_enc, queue_ids):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
