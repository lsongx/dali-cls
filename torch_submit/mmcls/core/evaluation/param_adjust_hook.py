from mmcv.runner import Hook


class ModelParamAdjustHook(Hook):
    """Adjust the param of a model (by epochs).
    """

    def __init__(self,
                 param_name_adjust_epoch_value,
                 logger=None):
        self.param_name_adjust_epoch_value = param_name_adjust_epoch_value
        self.logger = logger

    def before_train_epoch(self, runner):
        for name, adjust_epoch, value in self.param_name_adjust_epoch_value:
            if (runner.epoch+1) == adjust_epoch:
                setattr(runner.model.module, name, value)
                if self.logger:
                    self.logger.info(f'{name} is set to {value}')
