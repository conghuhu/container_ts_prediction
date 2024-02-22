from huawei.exp.exp_basic import Exp_Basic
from models.seqformer.DSFormer import DsFormer


class Exp_DsFormer(Exp_Basic):
    def __init__(self, args):
        super(Exp_DsFormer, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = DsFormer(args.timestep, args.feature_size, args.hidden_size, args.enc_layers,
                         args.num_heads,
                         args.ffn_hidden_size, args.dropout, args.pre_len,
                         args.use_RevIN, distil=args.conv)
        print(model)
        return model
