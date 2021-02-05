import logging
from collections import OrderedDict

import mmcv
from mmcv.runner import obj_from_dict
from mmcls.utils.checkpoint import load_checkpoint

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

import mmcls
from mmcls.core import auto_fp16, force_fp32
from mmcls.core.evaluation import accuracy

from .. import builder
from ..registry import CLASSIFIERS


@CLASSIFIERS.register_module
class Distill(nn.Module):
    """Base class for Classifiers"""

    def __init__(self, 
                 teacher_nets, 
                 student_net,
                 ce_loss, 
                 distill_loss, 
                 pretrained=None, 
                 backbone_init_cfg=None,
                 ce_loss_alpha=1,
                 distill_loss_alpha=0.5,
                 save_teacher_outputs=False,
                 load_teacher_outputs=False,
                 save_only_student=True):
        """
        """
        super(Distill, self).__init__()
        self.fp16_enabled = False
        self.teacher_nets = nn.ModuleList(
            [builder.build_backbone(teacher_net) for teacher_net in teacher_nets]) 
        self.student_net = builder.build_backbone(student_net)
        self.ce_loss = builder.build_loss(ce_loss)
        self.distill_loss = builder.build_loss(distill_loss)
        self.ce_loss_alpha = ce_loss_alpha
        self.distill_loss_alpha = distill_loss_alpha
        self.save_teacher_outputs = save_teacher_outputs
        self.load_teacher_outputs = load_teacher_outputs
        self.save_only_student = save_only_student
        self.init_weights(self.student_net, pretrained, backbone_init_cfg)
        if self.save_teacher_outputs:
            self.epoch_teacher_outputs = {}
        for t in self.teacher_nets:
            for param in t.parameters():
                param.requires_grad = False
            for module in t.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.momentum = 0

    @staticmethod
    def init_weights(net, pretrained, backbone_init_cfg):
        # even pretrained, still need init for eps
        if isinstance(backbone_init_cfg, str):
            initializer = getattr(mmcls.models.initializers, backbone_init_cfg)
            initializer(net)
        if pretrained is not None:
            logger = logging.getLogger()
            load_checkpoint(net, pretrained, map_location='cpu',
                            strict=False, logger=logger)
            logger.info('load model from: {}'.format(pretrained))

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, labels, return_loss=True):
        if return_loss:
            return self.forward_train(img, labels)
        else:
            return self.forward_test(img, labels)

    def forward_train(self, imgs, labels):
        with torch.no_grad():
            t_out = 0
            if self.load_teacher_outputs:
                out_list = self.epoch_teacher_outputs[imgs.sum().item()]
                for out in out_list:
                    out = out.to(imgs.device)
                    if self.distill_loss.with_soft_target:
                        out = out.softmax(dim=1)
                    t_out += out
                t_out /= len(self.teacher_nets)
            else:
                if self.save_teacher_outputs:
                    out_save = []
                for t in self.teacher_nets:
                    t.eval()
                    out = t(imgs)
                    if self.save_teacher_outputs:
                        out_save.append(out.cpu())
                    if self.distill_loss.with_soft_target:
                        out = out.softmax(dim=1)
                    t_out += out
                if self.save_teacher_outputs:
                    idx = imgs.sum().item()
                    self.epoch_teacher_outputs[idx] = out_save
                t_out /= len(self.teacher_nets)
        s_out = self.student_net(imgs)
        losses = self.get_loss(s_out, t_out, labels)
        return losses

    def forward_test(self, imgs, labels):
        outputs = self.student_net(imgs)
        return outputs

    @force_fp32(apply_to=('s_out', 't_out'))
    def get_loss(self, s_out, t_out, labels):
        losses = dict()
        losses['ce_loss'] = self.ce_loss(s_out, labels) * self.ce_loss_alpha
        losses['distill_loss'] = \
            self.distill_loss(s_out, t_out, labels) * self.distill_loss_alpha
        losses['s_acc'] = accuracy(s_out, labels)[0]
        losses['t_acc'] = accuracy(t_out, labels)[0]
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
        if self.save_only_student:
            return self.student_net.state_dict()
        return self.state_dict()
