import time

import torch


class Exp_Basic(object):
    def __init__(self, args):
        start = time.time()
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self._load_data()
        print("模型已初始化, 耗时{}s".format(time.time() - start))

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        raise NotImplementedError
        return None

    def _load_data(self):
        pass

    def _get_data(self, flag):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass
