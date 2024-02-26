from huawei.exp.exp_basic import Exp_Basic
from models.CNN_LSTM.pCNN_LSTM import PCNN_LSTM


class Exp_pCNN_LSTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_pCNN_LSTM, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = PCNN_LSTM(args.feature_size, args.output_size, args.hidden_size, args.num_layers,
                          args.pre_len)
        print(model)
        return model
