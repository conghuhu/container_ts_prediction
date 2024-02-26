import numpy as np
from torch import nn


class MultivariateMultiStepLSTM(nn.Module):
    def __init__(self, feature_size, hidden_layer_size, output_size, num_layers=1, bidirectional=False):
        super(MultivariateMultiStepLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=feature_size,
                            hidden_size=hidden_layer_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.lstm_output_dim = self.hidden_layer_size * 2 if bidirectional else self.hidden_layer_size

        # Fully Connected Layer for output
        self.linear = nn.Linear(self.lstm_output_dim, output_size)

        print("Number Parameters: lstm", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, input_seq, batch_idx):
        # input_seq shape: (B, T, F)
        lstm_out, _ = self.lstm(input_seq)

        # Select the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]

        # Pass through Linear layer, output shape (B, P)
        predictions = self.linear(last_time_step_out)

        # Reshape to (B, P, 1) to match the expected output size
        predictions = predictions.unsqueeze(-1)

        return predictions
