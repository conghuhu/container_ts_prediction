from huawei.exp.exp_basic import Exp_Basic
from models.Transformer.transformer import Transformer


class Exp_Transformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Transformer, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = Transformer(args.feature_size, args.hidden_size, args.enc_layers, args.dec_layers, args.num_heads,
                            args.dropout,
                            self.device, args.pre_len, args.timestep, args.output_size, args.use_RevIN)

        return model
