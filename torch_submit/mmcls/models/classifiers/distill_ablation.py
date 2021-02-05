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

from .distill import Distill
from .. import builder
from ..registry import CLASSIFIERS


@CLASSIFIERS.register_module
class DistillAblation(Distill):
    """Base class for Classifiers"""

    def forward_train(self, imgs, labels):
        s_out = self.student_net(imgs)
        # with torch.no_grad():
        #     s_loss = nn.functional.cross_entropy(s_out, labels, reduction='none')

        #     # t0_out = self.teacher_nets[0](imgs)
        #     # t0_loss = nn.functional.cross_entropy(t0_out, labels, reduction='none')
        #     # t0_s_margin = t0_loss - s_loss

        #     t1_out = self.teacher_nets[1](imgs)
        #     t1_loss = nn.functional.cross_entropy(t1_out, labels, reduction='none')
        #     t1_s_margin = t1_loss - s_loss

        #     # t_out = t1_out.softmax(dim=1)
        #     # mask = t1_s_margin > 0
        #     # t_out[mask] = t0_out.softmax(dim=1)[mask]

        #     # t_out = t0_out.softmax(dim=1)
        #     # mask = (t0_s_margin < 0) & (t1_s_margin < 0)
        #     # t_out[mask] = t1_out.softmax(dim=1)[mask]

        #     # t_out = t1_out.softmax(dim=1)
        #     # mask = t1_s_margin > 0
        #     # one_hot_target = torch.zeros_like(t_out)
        #     # one_hot_target.scatter_(1, labels[:,None], 1)
        #     # t_out[mask] = one_hot_target[mask]

        #     t_out = t1_out.softmax(dim=1)
        #     mask = t1_s_margin < 0
        #     one_hot_target = torch.zeros_like(t_out)
        #     one_hot_target.scatter_(1, labels[:,None], 1)
        #     t_out[mask] = one_hot_target[mask]

        #     percentage = mask.sum().float() / imgs.shape[0]
        t_out = torch.zeros_like(s_out)
        t_out.scatter_(1, labels[:,None], 1)
        t_out = t_out*0.9 + 0.9/1000

        losses = self.get_loss(s_out, t_out, labels)
        # losses['percentage'] = percentage
        return losses

