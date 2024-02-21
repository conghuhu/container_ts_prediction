from huawei.exp.exp_basic import Exp_Basic
from models.seqformer.seqformer import SeqFormer


class Exp_SeqFormer(Exp_Basic):
    def __init__(self, args):
        super(Exp_SeqFormer, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = SeqFormer(args.timestep, args.feature_size, args.hidden_size, args.enc_layers, args.dec_layers,
                          args.num_heads,
                          args.ffn_hidden_size, args.dropout, args.pre_norm, args.output_size, args.pre_len,
                          args.use_RevIN)

        return model
