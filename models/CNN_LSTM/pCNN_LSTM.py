import numpy as np
import torch
from torch import nn


class PCNN_LSTM(nn.Module):
    def __init__(self, feature_size, output_features, hidden_size, num_layers, pred_length):
        super(PCNN_LSTM, self).__init__()

        # 卷积层参数
        self.conv1 = nn.Conv1d(feature_size, hidden_size, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(feature_size, hidden_size, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(feature_size, hidden_size, kernel_size=3, dilation=4, padding=4)

        # LSTM层参数
        self.lstm = nn.LSTM(hidden_size * 3, hidden_size, num_layers, batch_first=True)

        # 线性层参数
        self.linear = nn.Linear(hidden_size, output_features)

        # 预测长度
        self.pred_length = pred_length

        print("Number Parameters: lstm", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, queue_ids):
        # x shape: [B, T, F] -> 需要调整为卷积层适应的形状 [B, F, T]
        x = x.permute(0, 2, 1)

        # 通过三个并行的卷积层
        conv1_out = torch.relu(self.conv1(x))
        conv2_out = torch.relu(self.conv2(x))
        conv3_out = torch.relu(self.conv3(x))

        # 合并卷积层的输出
        conv_out = torch.cat((conv1_out, conv2_out, conv3_out), 1)

        # 调整形状适应LSTM输入 [B, T, F]
        conv_out = conv_out.permute(0, 2, 1)

        # 通过LSTM层
        lstm_out, _ = self.lstm(conv_out)

        # Select the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]

        # Pass through Linear layer, output shape (B, P)
        predictions = self.linear(last_time_step_out)

        # Reshape to (B, P, 1) to match the expected output size
        predictions = predictions.unsqueeze(-1)

        return predictions
