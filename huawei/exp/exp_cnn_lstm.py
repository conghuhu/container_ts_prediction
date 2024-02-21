from huawei.exp.exp_basic import Exp_Basic
from models.CNN_LSTM.CNN_LSTM_net import CNN_LSTM_Attention


class Exp_CNN_LSTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_CNN_LSTM, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = CNN_LSTM_Attention(args.feature_size, args.hidden_size, args.enc_layers,
                                   args.out_channels, args.num_heads, args.output_size, args.bidirectional,
                                   args.dropout)
        print(model)
        return model
