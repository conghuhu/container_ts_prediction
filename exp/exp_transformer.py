from exp.exp_basic import Exp_Basic
from models.Transformer.transformer import Transformer


class Exp_Transformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Transformer, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = Transformer(args)

        return model
