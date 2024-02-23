from huawei.exp.exp_basic import Exp_Basic
from models.Transformer.FEDFormer import FEDFormer


class Exp_FEDFormer(Exp_Basic):
    def __init__(self, args):
        super(Exp_FEDFormer, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = FEDFormer(args)
        print(model)
        return model
