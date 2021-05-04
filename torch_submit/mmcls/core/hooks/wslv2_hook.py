import shutil
import os
import torch
import mmcv
from mmcv.runner import Hook

import mmcls
from mmcls.models import builder


class WSLv2Hook(Hook):
    """
    """

    def __init__(self, 
                 switch_epoch, 
                 optimizer_cfg, 
                 lr_config, 
                 loss, 
                 teacher_nets=None, 
                 logger=None):
        self.switch_epoch = switch_epoch
        self.optimizer_cfg = optimizer_cfg
        self.lr_config = lr_config
        self.loss = loss
        self.teacher_nets = teacher_nets
        self.logger = logger

    def before_train_epoch(self, runner):
        if (runner.epoch+1) != self.switch_epoch:
            return

        if self.teacher_nets is not None:
            runner.model.module.teacher_nets = torch.nn.ModuleList(
                [builder.build_backbone(teacher_net) for teacher_net in self.teacher_nets]).cuda()

        optimizer = mmcv.runner.build_optimizer(runner.model, self.optimizer_cfg)
        runner.optimizer = optimizer
        for idx, h in enumerate(runner._hooks):
            if isinstance(h, mmcv.runner.hooks.lr_updater.LrUpdaterHook):
                self.logger.info(f'Hook {runner._hooks[idx]} deleted')
                del runner._hooks[idx]
        lr_config = mmcv.runner.obj_from_dict(self.lr_config, mmcls.core.hooks)
        lr_config.before_run(runner)
        runner.register_lr_hook(lr_config)
        self.logger.info(f'Current hooks:\n {runner._hooks}')
        # change loss
        runner.model.module.ce_loss_alpha = 0
        runner.model.module.distill_loss = mmcls.models.builder.build_loss(self.loss) 
        self.logger.info('WSLv2Hook is added')
        if runner.rank == 0:
            if os.path.isfile(f'./data/out/epoch_{self.switch_epoch-1}.pth'):
                shutil.copy(f'./data/out/epoch_{self.switch_epoch-1}.pth', 
                            f'./data/out/switch_save.pth')

        for h in runner._hooks:
            if isinstance(h, mmcv.runner.hooks.optimizer.Fp16OptimizerHook):
                h.before_run(runner)
