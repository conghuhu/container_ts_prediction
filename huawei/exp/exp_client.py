from huawei.exp.exp_basic import Exp_Basic
from models.Transformer.Client import Client


class Exp_Client(Exp_Basic):
    def __init__(self, args):
        super(Exp_Client, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = Client(args.timestep, args.feature_size, args.pre_len, args.enc_layers,
                       args.num_heads, args.factor, args.dropout, args.d_ff, use_RevIN=args.use_RevIN)
        print(model)
        return model
