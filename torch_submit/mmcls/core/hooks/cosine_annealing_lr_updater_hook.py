from mmcv.runner.hooks.lr_updater import LrUpdaterHook, annealing_cos


class CosineAnnealingLrUpdaterHook(LrUpdaterHook):

    def __init__(self, max_progress, base_progress=0, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.max_progress = max_progress
        self.base_progress = base_progress
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(CosineAnnealingLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch-self.base_progress
        else:
            progress = runner.iter-self.base_progress
        max_progress = self.max_progress

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        return annealing_cos(base_lr, target_lr, progress / max_progress)
