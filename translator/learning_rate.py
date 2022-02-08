"""Learning rate schedules"""


class WarmUpLr:
    def __init__(self, warmup_steps, cte):
        self.warmup_steps = warmup_steps
        self.cte = cte**(-1/2)
        self.lr = None

    def calc_lr(self, it):
        lr = min(it**(-1/2), self.warmup_steps**(-3/2)*it)
        return self.cte*lr

    def __call__(self, it, param_groups):
        self.lr = self.calc_lr(it)
        for param_group in param_groups:
            param_group['lr'] = self.lr


class ConstantLr(float):
    def __call__(self, it, param_groups):
        if it == 1:
            for param_group in param_groups:
                param_group['lr'] = self
        pass


def wrap_lr(lr):
    if isinstance(lr, float):
        return ConstantLr(lr)
    return lr
