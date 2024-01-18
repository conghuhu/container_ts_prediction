import torch.nn.functional as F
from torch import nn


# 定义CNN + LSTM + Attention网络
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, out_channels, num_heads, output_size, bidirectional=False,
                 dropout=0.1):
        super(CNN_LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 卷积层
        self.conv1d = nn.Conv1d(in_channels=feature_size, out_channels=out_channels, kernel_size=3, padding=1)

        # LSTM层
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)

        self.lstm_output_dim = self.hidden_size * 2 if bidirectional else self.hidden_size

        # 注意力层
        # 如果batch_first为True，则输出的shape为[batch_size, timestep, hidden_size]
        self.attention = nn.MultiheadAttention(embed_dim=self.lstm_output_dim, num_heads=num_heads, batch_first=True,
                                               dropout=dropout)

        # 输出层
        self.fc = nn.Linear(self.lstm_output_dim, output_size)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        # 原始x：B, T, feature_size
        # 卷积层期望的输入维度是 [B, feature_size, T]，因此需要转置
        x = x.transpose(1, 2)

        # 卷积运算
        output = F.relu(self.conv1d(x))

        batch_size = x.shape[0]

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).fill_(
                0).float()
            c_0 = x.data.new(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).fill_(
                0).float()
        else:
            h_0, c_0 = hidden

        # 再次转置为LSTM需要的格式 [B, T, hidden_dim]
        output = output.transpose(1, 2)  # B, feature_size, T

        # LSTM运算
        output, (h_0, c_0) = self.lstm(output, (h_0, c_0))  # size(B, T, D*hidden_size)

        # 注意力计算
        attention_output, attn_output_weights = self.attention(output, output, output)  # 输出维度: [B, T, lstm_output_dim]

        # 取最后一个时间步的输出
        output = attention_output[:, -1, :]

        # 全连接层
        output = self.fc(output)  # 维度: [B, P]

        return output.unsqueeze(-1)
