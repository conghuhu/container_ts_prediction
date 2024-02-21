from huawei.exp.exp_basic import Exp_Basic
from models.PatchTST.PatchTST import PatchTST


class Exp_PatchTST(Exp_Basic):
    def __init__(self, args):
        super(Exp_PatchTST, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = PatchTST(args.timestep, args.feature_size, args.pre_len, args.hidden_size, args.enc_layers,
                         args.dropout, args.factor, args.num_heads,
                         args.output_attention, args.d_ff,
                         args.activation, args.use_RevIN)

        return model
