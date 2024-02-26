import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class PCNN_LSTM(nn.Module):
    def __init__(self, feature_size, output_size, pre_len, num_layers, num_channels, kernel_size=2, dropout=0.2,
                 device=torch.device('cpu')):
        super(PCNN_LSTM, self).__init__()
        layers = []
        self.pre_len = pre_len
        self.n_layers = num_layers
        self.hidden_size = num_channels[-2]
        self.device = device

        self.hidden = nn.Linear(num_channels[-1], num_channels[-2])
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers, bias=True,
                            batch_first=True)  # output (batch_size, obs_len, hidden_size)
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = feature_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-2], output_size)

        print("Number Parameters: pCNN-LSTM", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, queue_ids):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)

        batch_size, obs_len, features_size = x.shape  # (batch_size, obs_len, features_size)
        xconcat = self.hidden(x)  # (batch_size, obs_len, hidden_size)
        H = torch.zeros(batch_size, obs_len - 1, self.hidden_size).to(
            self.device)  # (batch_size, obs_len-1, hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(
            self.device)  # (num_layers, batch_size, hidden_size)
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)  # (batch_size, 1, hidden_size)
            out, (ht, ct) = self.lstm(xt, (ht, ct))  # ht size (num_layers, batch_size, hidden_size)
            htt = ht[-1, :, :]  # (batch_size, hidden_size)
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)  # (batch_size, obs_len-1, hidden_size)

        x = self.linear(H)
        return x[:, -self.pre_len:, 0:1]
