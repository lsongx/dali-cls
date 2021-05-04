import random
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
from mmcls.utils import get_root_logger

from .. import builder
from ..registry import CLASSIFIERS


def get_output_channels(module_layer):
    if isinstance(module_layer, nn.Conv2d):
        return module_layer.out_channels
    child_modules = list(module_layer.children())
    if len(child_modules) > 1:
        for child in child_modules[::-1]:
            if isinstance(child, nn.Conv2d):
                return child.out_channels
    raise NotImplementedError


def expand_connect_index(connect_index, end_index):
    """Expand the connect index list [a,b] to [(0,a),(a,b),(b,end)]
    """
    out_idx = []
    last_idx = 0
    for idx in connect_index:
        out_idx.append((last_idx, idx))
        last_idx = idx
    out_idx.append((last_idx, end_index))
    return out_idx


def adjust_bn_tracking(model, mode):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            # module.track_running_stats = mode
            # --------------- change the track_running_stats will not work
            # track_running_stats is designed as intrinsic property of the layer
            # not a dynamic status variable
            if mode:
                module.momentum = 0.1
            else:
                module.momentum = 0


@CLASSIFIERS.register_module
class HybridTransformer(nn.Module):
    """Base class for Classifiers"""

    def __init__(self, 
                 teacher_net, 
                 student_net,
                 loss, 
                 teacher_connect_index,
                 student_connect_index,
                 teacher_pretrained=None,
                 student_pretrained=None,
                 teacher_channels=None,
                 student_channels=None,
                 teacher_backbone_init_cfg=None,
                 student_backbone_init_cfg=None,
                 ori_net_path_loss_alpha=0.5,
                 random_hybrid=1,
                 save_only_student=False):
        """teacher_channels, student_channels are the channels 
        of the connecting node.
        """
        super().__init__()
        self.fp16_enabled = False
        self.teacher_net = builder.build_backbone(teacher_net)
        self.student_net = builder.build_backbone(student_net)
        self.loss = builder.build_loss(loss)
        self.teacher_connect_index = teacher_connect_index
        self.student_connect_index = student_connect_index
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels
        self.ori_net_path_loss_alpha = ori_net_path_loss_alpha
        self.save_only_student = save_only_student
        self.use_random_hybrid_index = isinstance(self.student_connect_index, list)
        self.random_hybrid = random_hybrid

        self.init_weights(self.teacher_net, teacher_pretrained, 
                          teacher_backbone_init_cfg)
        self.init_weights(self.student_net, student_pretrained, 
                          student_backbone_init_cfg)
        self.init_connect_module_list()
        for param in self.teacher_net.parameters():
            param.requires_grad = False
        
        self.teacher_net_list = nn.Sequential(*list(self.teacher_net.children()))
        self.student_net_list = nn.Sequential(*list(self.student_net.blocks.children()))

    @staticmethod
    def init_weights(net, pretrained, backbone_init_cfg):
        # even pretrained, still need init for eps
        if isinstance(backbone_init_cfg, str):
            initializer = getattr(mmcls.models.initializers, backbone_init_cfg)
            initializer(net.ori_net)
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(net.ori_net, pretrained, map_location='cpu',
                            strict=False, logger=logger)
            logger.info('load model from: {}'.format(pretrained))

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, labels, return_loss=True):
        if return_loss:
            return self.forward_train(img, labels)
        else:
            return self.forward_test(img, labels)

    def forward_student_remain(self, x):
        x = self.student_net.norm(x)[:, 0]
        x = self.student_net.pre_logits(x)
        x = self.student_net.head(x)
        return x

    def forward_train(self, imgs, labels):
        with torch.no_grad():
            t_feat = self.teacher_net_list[:self.teacher_connect_index](imgs)
            b, c, h, w = t_feat.shape
        # timm 0.4.5
        x = imgs
        x = self.student_net.patch_embed(x)
        cls_tokens = self.student_net.cls_token.expand(b, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.student_net.pos_embed
        x = self.student_net.pos_drop(x)

        if self.use_random_hybrid_index:
            sc = random.sample(self.student_connect_index, 1)[0]
        else:
            sc = self.student_connect_index

        x = self.student_net_list[:sc](x)
        s_token = x[:, 0]
        s_feat = x[:, 1:]

        s_out = self.forward_student_remain(self.student_net_list[sc:](x))
        if random.random() < self.random_hybrid:
            s2t_feat = self.s2t(s_feat.permute([0,2,1]).reshape([b,-1,h,w]))
            s2t_out = self.teacher_net_list[self.teacher_connect_index:](s2t_feat)
            t2s_feat = torch.cat(
                [s_token[:,:,None], self.t2s(t_feat).reshape([b,-1,h*w])], dim=2).permute([0,2,1])
            t2s_out = self.forward_student_remain(self.student_net_list[sc:](t2s_feat))
        else:
            s2t_out, t2s_out = None, None

        losses = self.get_loss(s2t_out, t2s_out, s_out, labels)
        return losses

    def forward_test(self, imgs, labels):
        outputs = self.student_net(imgs)
        return outputs

    @force_fp32(apply_to=('s2t_out', 't2s_out', 's_out',))
    def get_loss(self, s2t_out, t2s_out, s_out, labels):
        losses = dict()
        if self.ori_net_path_loss_alpha<1 and s2t_out is not None:
            losses['s2t_loss'] = \
                self.loss(s2t_out, labels)*(1-self.ori_net_path_loss_alpha)
            losses['t2s_loss'] = \
                self.loss(t2s_out, labels)*(1-self.ori_net_path_loss_alpha)
            losses['s2t_acc'] = accuracy(s2t_out, labels)[0]
            losses['t2s_acc'] = accuracy(t2s_out, labels)[0]
        losses['s_loss'] = \
            self.loss(s_out, labels) * self.ori_net_path_loss_alpha
        losses['s_acc'] = accuracy(s_out, labels)[0]
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

    def init_connect_module_list(self):
        self.t2s = nn.Sequential(
            nn.Conv2d(self.teacher_channels, self.student_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.student_channels),
            nn.ReLU(),
            nn.Conv2d(self.student_channels, self.student_channels, 3, 1, 1),
        )
        self.s2t = nn.Sequential(
            nn.Conv2d(self.student_channels, self.teacher_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.teacher_channels),
            nn.ReLU(),
            nn.Conv2d(self.teacher_channels, self.teacher_channels, 3, 1, 1),
        )
