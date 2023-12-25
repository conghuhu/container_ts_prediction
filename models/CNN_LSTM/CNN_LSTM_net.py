from torch import nn


# 定义CNN + LSTM + Attention网络
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, feature_size, timestep, hidden_size, num_layers, out_channels, num_heads, output_size):
        super(CNN_LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 卷积层
        self.conv1d = nn.Conv1d(in_channels=feature_size, out_channels=out_channels, kernel_size=3, padding=1)

        # LSTM层
        self.lstm = nn.LSTM(out_channels, hidden_size, num_layers, batch_first=True)

        # 注意力层
        # 如果batch_first为True，则输出的shape为[batch_size, timestep, hidden_size]
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_heads, batch_first=True,
                                               dropout=0.8)

        # 输出层
        self.fc1 = nn.Linear(timestep * hidden_size, 256)
        self.fc2 = nn.Linear(256, output_size)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        # 原始x：batch_size, timestep, feature_size
        x = x.transpose(1, 2)  # batch_size, feature_size, timestep

        # 卷积运算
        output = self.conv1d(x)

        batch_size = x.shape[0]

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        output = output.transpose(1, 2)  # batch_size, timestep, feature_size,

        # LSTM运算
        output, (h_0, c_0) = self.lstm(output, (h_0, c_0))  # size(batchSize, seq_len, D*hidden_size)

        # 注意力计算
        attention_output, attn_output_weights = self.attention(output, output, output)

        # 使用注意力层的输出
        output = attention_output.flatten(start_dim=1)

        # 全连接层
        output = self.fc1(output)
        output = self.relu(output)

        output = self.fc2(output)

        return output
