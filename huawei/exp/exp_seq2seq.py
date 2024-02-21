from huawei.exp.exp_basic import Exp_Basic
from models.seq2seq.en_dn_wrapper_net import EncoderDecoderWrapper


class Exp_Seq2Seq(Exp_Basic):
    def __init__(self, args):
        super(Exp_Seq2Seq, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = EncoderDecoderWrapper(args.feature_size, args.output_size, args.hidden_size, args.enc_layers,
                                      args.pre_len, args.timestep, args.bidirectional)
        return model
