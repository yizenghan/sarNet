from mmcv.runner.hooks import LrUpdaterHook
import math


class CosineLrUpdaterHook(LrUpdaterHook):

    def __init__(self, t_max, eta_min=0, **kwargs):
        assert isinstance(t_max, int)
        self.t_max = t_max
        self.eta_min = eta_min
        super(CosineLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress / self.t_max)) / 2

