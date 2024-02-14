from exp.exp_basic import Exp_Basic
from models.Transformer.DsFormer import DsFormer


class Exp_DsFormer(Exp_Basic):
    def __init__(self, args):
        super(Exp_DsFormer, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = DsFormer(args.feature_size, args.hidden_size, args.num_heads, args.ffn_hidden_size,
                         args.num_layers,
                         args.queue_embed_dim,
                         self.device, args.pre_len, args.timestep, args.dropout)

        return model
