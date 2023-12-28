import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为8
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # LSTM运算
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))

        return output, h_0, c_0


class Decoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(Decoder, self).__init__()
        self.output_size = output_size  # 输出的时间步
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数
        # feature_size为特征维度，就是每个时间点对应的特征数量
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, h_0, c_0):
        # 打标注
        x = torch.zeros(h_0.shape[1], self.output_size, self.hidden_size).to(h_0.device)

        # LSTM运算
        output, _ = self.lstm(x, (h_0, c_0))

        output = self.fc(output)

        return output[:, -1, :]


class Seq2Seq(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(feature_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, hidden_size, num_layers, output_size)

    def forward(self, x):
        # x.shape(batch_size * timeStep * feature_size)
        _, h_n, c_n = self.encoder(x)

        output = self.decoder(h_n, c_n)
        # output size(batch_size * output_size)
        return output
