import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class DynamicEmbedding(nn.Module):
    def __init__(self, embed_dim, max_size=10000, padding_idx=0):
        super(DynamicEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_size = max_size
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.Tensor(max_size, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].zero_()

    def forward(self, input):
        # Expand embedding weight if input exceeds current max size
        max_idx = input.max().item()
        if max_idx >= self.max_size:
            extra_size = max_idx + 1 - self.max_size
            extra_weight = nn.Parameter(torch.Tensor(extra_size, self.embed_dim))
            nn.init.normal_(extra_weight)
            self.weight = nn.Parameter(torch.cat([self.weight, extra_weight], 0))
            self.max_size = max_idx + 1
        return nn.functional.embedding(input, self.weight, self.padding_idx)

    def extra_repr(self):
        return 'embed_dim={}, max_size={}, padding_idx={}'.format(
            self.embed_dim, self.max_size, self.padding_idx
        )


class DsFormer(nn.Module):
    def __init__(self, feature_size, hidden_size, nhead, ffn_hidden_size, num_layers, queue_embed_dim, device, pre_len,
                 timestep,
                 dropout=0.5,
                 max_queues=1000):
        """
        :param feature_size: 这是输入特征的大小。在时间序列预测的上下文中，这通常是特征数量（不包括QUEUE_ID）。如果你的输入特征经过独热编码或归一化，ntoken应该等于转换后的特征维度。
        :param hidden_size: 这是模型内部的嵌入大小。它定义了模型中间层的大小，以及时间序列特征经过嵌入层转换后的维度。
        :param nhead: Transformer模型中多头注意力机制的头数。多头注意力允许模型同时关注序列的不同位置，nhead通常是能被ninp整除的数。
        :param ffn_hidden_size: 这是Transformer内部的前馈网络（Feedforward Neural Network, FNN）的大小。FNN在每个注意力层后面用于进一步处理数据。
        :param num_layers: 这是Transformer模型堆叠的编码器层数。增加层数可以提高模型的复杂度和学习能力，但也会增加计算成本和过拟合的风险。
        :param queue_embed_dim: 这是QUEUE_ID嵌入向量的维度。它定义了每个QUEUE_ID转换后的向量大小，这可以帮助模型学习到关于不同QUEUE_ID（服务）的细微差异。
        :param device:
        :param pre_len:
        :param timestep:
        :param dropout: 这是模型中使用的dropout率，用于正则化和防止过拟合。Dropout在训练过程中随机丢弃一定比例的节点，以增加模型的泛化能力。
        :param max_queues:
        """
        super(DsFormer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size + queue_embed_dim, nhead, ffn_hidden_size, dropout,
                                                    device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.feature_encoder = nn.Embedding(feature_size, hidden_size)
        self.queue_embedding = DynamicEmbedding(queue_embed_dim, max_size=max_queues)
        self.hidden_size = hidden_size
        self.decoder = nn.Linear(hidden_size + queue_embed_dim, 1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + queue_embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pre_len)
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.feature_encoder.weight.data.uniform_(-initrange, initrange)
        self.queue_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Extract and remove the queue_id from src
        queue_ids = src[:, :, 2].long()  # Assuming queue_id is the 3rd feature and integer
        src = torch.cat((src[:, :, :2], src[:, :, 3:]), dim=2)  # Remove the queue_id from the features

        queue_embed = self.queue_embedding(queue_ids).unsqueeze(1).repeat(1, src.size(1),
                                                                          1)  # Repeat embedding for each time step
        src = torch.cat((src, queue_embed), dim=2)

        src = self.feature_encoder(src) * math.sqrt(self.hidden_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output
