from exp.exp_basic import Exp_Basic
from models.CNN_LSTM.LSTM import MultivariateMultiStepLSTM


class Exp_LSTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_LSTM, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = MultivariateMultiStepLSTM(args.feature_size, args.hidden_size, args.output_size, args.enc_layers,
                                          args.bidirectional)
        print(model)
        return model
