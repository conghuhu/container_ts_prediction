from exp.exp_basic import Exp_Basic
from models.DLinear.DLinear import DLinear


class Exp_DLinear(Exp_Basic):
    def __init__(self, args):
        super(Exp_DLinear, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = DLinear(args.timestep, args.pre_len, args.moving_avg, args.enc_inc, args.individual)
        return model
