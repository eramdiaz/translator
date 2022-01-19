"""Learning rate schedules"""


class WarmUpLr:
    def __init__(self, warmup_steps, cte, param_groups):
        self.warmup_steps = warmup_steps
        self.cte = cte**(-1/2)
        self.param_groups = param_groups
        self.lr = None

    def calc_lr(self, it):
        lr = min(it**(-1/2), self.warmup_steps**(-3/2)*it)
        return self.cte*lr

    def __call__(self, it):
        self.lr = self.calc_lr(it)
        for param_group in self.param_groups:
            param_group['lr'] = self.lr
