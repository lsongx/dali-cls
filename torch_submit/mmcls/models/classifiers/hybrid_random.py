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
class HybridRandom(nn.Module):
    """Base class for Classifiers"""

    def __init__(self, 
                 teacher_net, 
                 student_net,
                 loss, 
                 teacher_connect_index,
                 student_connect_index,
                 teacher_pretrained, 
                 student_pretrained=None, 
                 teacher_channels=None,
                 student_channels=None,
                 teacher_backbone_init_cfg=None,
                 student_backbone_init_cfg=None,
                 ori_net_path_loss_alpha=0.5,
                 switch_prob=0.5,
                 save_only_student=False):
        """teacher_channels, student_channels are the channels 
        of the connecting node.
        """
        super().__init__()
        self.fp16_enabled = False
        self.teacher_net = builder.build_backbone(teacher_net)
        self.student_net = builder.build_backbone(student_net)
        self.loss = builder.build_loss(loss)
        assert len(teacher_connect_index) == len(student_connect_index)
        self.teacher_connect_index = teacher_connect_index
        self.student_connect_index = student_connect_index
        self.ori_net_path_loss_alpha = ori_net_path_loss_alpha
        self.switch_prob = switch_prob
        self.save_only_student = save_only_student

        self.init_conn_channels(teacher_channels, student_channels)
        self.init_weights(self.teacher_net, teacher_pretrained, 
                          teacher_backbone_init_cfg)
        self.init_weights(self.student_net, student_pretrained, 
                          student_backbone_init_cfg)
        self.init_connect_module_list()
        for param in self.teacher_net.parameters():
            param.requires_grad = False

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

    def forward_train(self, imgs, labels):
        t_out_line = imgs
        s_out_line = imgs
        if self.ori_net_path_loss_alpha < 1:
            adjust_bn_tracking(self.student_net, False)
            for idx, (t, s) in enumerate(zip(self.t_idx, self.s_idx)):
                # swith feature map
                if np.random.random() > self.switch_prob:
                    t_out_line, s_out_line = s_out_line, t_out_line
                    if idx > 0 and not self.ignore_conn_mask[idx-1]:
                        t_out_line = self.s2t_conv_list[idx-1](t_out_line)
                        s_out_line = self.t2s_conv_list[idx-1](s_out_line)
                t_out_line = \
                    self.teacher_net.sequence_warp[t[0]:t[1]](t_out_line)
                s_out_line = \
                    self.student_net.sequence_warp[s[0]:s[1]](s_out_line)
            adjust_bn_tracking(self.student_net, True)
        else:
            t_out_line = None
            s_out_line = None
        s_out = self.student_net(imgs)
        losses = self.get_loss(t_out_line, s_out_line, s_out, labels)
        return losses

    def forward_test(self, imgs, labels):
        outputs = self.student_net(imgs)
        return outputs

    @force_fp32(apply_to=('t_out_line', 's_out_line', 's_out',))
    def get_loss(self, t_out_line, s_out_line, s_out, labels):
        losses = dict()
        if self.ori_net_path_loss_alpha < 1:
            losses['t_line_loss'] = \
                self.loss(t_out_line, labels)*(1-self.ori_net_path_loss_alpha)
            losses['s_line_loss'] = \
                self.loss(s_out_line, labels)*(1-self.ori_net_path_loss_alpha)
            losses['t_line_acc'] = accuracy(t_out_line, labels)[0]
            losses['s_line_acc'] = accuracy(s_out_line, labels)[0]
        # losses['s_loss'] = \
        #     self.loss(s_out, labels) * self.ori_net_path_loss_alpha
        losses['s_loss'] = self.loss(s_out, labels)
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

    def init_conn_channels(self, teacher_channels, student_channels):
        def init_channel(channels, net, index):
            if channels is not None:
                return channels
            out = []
            for i in index:
                out.append(get_output_channels(net.sequence_warp[i-1]))
            return out
        self.teacher_channels = init_channel(teacher_channels, 
                                             self.teacher_net, 
                                             self.teacher_connect_index)
        self.student_channels = init_channel(student_channels, 
                                             self.student_net, 
                                             self.student_connect_index)

    def init_connect_module_list(self):
        self.ignore_conn_mask = []
        self.t2s_conv_list = nn.ModuleList()
        self.s2t_conv_list = nn.ModuleList()
        for tc, sc in zip(self.teacher_channels, self.student_channels):
            if tc == sc:
                self.ignore_conn_mask.append(True)
                continue
            self.ignore_conn_mask.append(False)
            self.t2s_conv_list.append(nn.Conv2d(tc, sc, 1))
            self.s2t_conv_list.append(nn.Conv2d(sc, tc, 1))
        self.t_idx = expand_connect_index(self.teacher_connect_index, 
                                          len(self.teacher_net.sequence_warp))
        self.s_idx = expand_connect_index(self.student_connect_index,
                                          len(self.student_net.sequence_warp))
