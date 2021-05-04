from collections import OrderedDict

import torch
import mmcv
from mmcv.runner import obj_from_dict
from mmcls.utils.checkpoint import load_checkpoint

import numpy as np
import torch.nn as nn
import torch.distributed as dist

import mmcls
from mmcls.core import auto_fp16, force_fp32
from mmcls.core.evaluation import accuracy
from mmcls.utils import get_root_logger

from .. import builder
from ..registry import CLASSIFIERS



@CLASSIFIERS.register_module
class BaseClassifier(nn.Module):
    """Base class for Classifiers"""

    def __init__(self, 
                 backbone, 
                 loss, 
                 pretrained=None, 
                 backbone_init_cfg=None):
        super(BaseClassifier, self).__init__()
        self.fp16_enabled = False
        self.backbone = builder.build_backbone(backbone)
        self.loss = builder.build_loss(loss)
        self.init_weights(pretrained, backbone_init_cfg)

    def init_weights(self, pretrained, backbone_init_cfg):
        # even pretrained, still need init for eps
        if isinstance(backbone_init_cfg, str):
            initializer = getattr(mmcls.models.initializers, backbone_init_cfg)
            initializer(self.backbone)
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self.backbone, pretrained, map_location='cpu',
                            strict=False, logger=logger)
            logger.info('load model from: {}'.format(pretrained))

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, labels, return_loss=True):
        if return_loss:
            return self.forward_train(img, labels)
        else:
            return self.forward_test(img, labels)

    def forward_train(self, imgs, labels):
        # dist.barrier()
        torch.cuda.synchronize()
        outputs = self.backbone(imgs)
        losses = self.get_loss(outputs, labels)
        return losses

    def forward_test(self, imgs, labels):
        outputs = self.backbone(imgs)
        return outputs

    @force_fp32(apply_to=('outputs', ))
    def get_loss(self, outputs, labels):
        losses = dict()
        losses['loss'] = self.loss(outputs, labels)
        losses['acc'] = accuracy(outputs, labels)[0]
        return losses

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        x = data[0]["data"]
        y = data[0]["label"].squeeze().cuda().long()
        losses = self(x, y)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=int(x.shape[0]))

        return outputs

    def val_step(self, data, optimizer):
        x = data[0]["data"]
        y = data[0]["label"].squeeze().cuda().long()
        losses = self(x, y)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=int(x.shape[0]))

        return outputs

    def get_model(self):
        return self.backbone.state_dict()
