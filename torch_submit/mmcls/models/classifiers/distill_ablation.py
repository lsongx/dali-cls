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
        with torch.no_grad():
            for t in self.teacher_nets:
                t.eval()
            s_loss = nn.functional.cross_entropy(s_out, labels, reduction='none')

            # t0_out = self.teacher_nets[0](imgs)
            # t0_loss = nn.functional.cross_entropy(t0_out, labels, reduction='none')
            # t0_s_margin = t0_loss - s_loss

            t1_out = self.teacher_nets[1](imgs)
            t1_loss = nn.functional.cross_entropy(t1_out, labels, reduction='none')
            t1_s_margin = t1_loss - s_loss

            # t_out = t1_out.softmax(dim=1)
            # mask = t1_s_margin > 0
            # t_out[mask] = t0_out.softmax(dim=1)[mask]

            # t_out = t0_out.softmax(dim=1)
            # mask = (t0_s_margin < 0) & (t1_s_margin < 0)
            # t_out[mask] = t1_out.softmax(dim=1)[mask]

            # t_out = t1_out.softmax(dim=1)
            # mask = t1_s_margin > 0
            # one_hot_target = torch.zeros_like(t_out)
            # one_hot_target.scatter_(1, labels[:,None], 1)
            # t_out[mask] = one_hot_target[mask]

            # t_out = t1_out.softmax(dim=1)
            # mask = t1_s_margin < 0 # t1 better than s
            # soft_target = torch.zeros_like(t_out)
            # soft_target.scatter_(1, labels[:,None], 1)
            # soft_target = soft_target*0.9+0.1/1000
            # t_out[mask] = soft_target[mask]

            t_out = (t1_out*4).softmax(dim=1)
            better_mask = t1_s_margin < 0 # t1 better than s
            worse_mask = ~better_mask
            t_wrong_mask = t_out.max(1).indices!=labels
            s_wrong_mask = s_out.max(1).indices!=labels
            t_s_diff_mask = t_out.max(1).indices!=s_out.max(1).indices

            target = torch.zeros_like(t_out)
            target = target.scatter_(1, labels[:,None], 1).bool()
            # soft_target = target.to(t_out.dtype)*0.9+0.1/1000
            soft_target = target.to(t_out.dtype)*0.99+0.01/1000
            t_target = torch.zeros_like(t_out)
            t_target = t_target.scatter_(1, t_out.max(1).indices[:,None], 1).bool()
            s_target = torch.zeros_like(t_out)
            s_target = s_target.scatter_(1, s_out.max(1).indices[:,None], 1).bool()

            self.temperature = 1
            student_softmax_with_t = (s_out/self.temperature).softmax(dim=1)
            student_softmax = s_out.softmax(dim=1)
            teacher_softmax_with_t = (t1_out/self.temperature).softmax(dim=1)

            bias = student_softmax[target] - 1
            ts_diff = student_softmax_with_t - teacher_softmax_with_t
            var = self.temperature*ts_diff[target] - bias
            # mask = worse_mask
            # mask = worse_mask & (~t_s_diff_mask) & t_wrong_mask & s_wrong_mask
            # mask = worse_mask & (~t_s_diff_mask) & t_wrong_mask & s_wrong_mask
            # mask = worse_mask & t_s_diff_mask & t_wrong_mask & s_wrong_mask
            # mask = worse_mask & (~t_wrong_mask) & (~s_wrong_mask) 
            mask = (var.abs()>bias.abs()) & (~t_wrong_mask) & (~s_wrong_mask) 

            logs = {}
            logs['bias'] = bias.mean()
            logs['var'] = var.mean()
            logs['vb'] = (var+bias).mean()
            logs['vbmask'] = (var.abs()>bias.abs()).float().mean()

            if mask.sum()>0:
                # t0_mask_out = self.teacher_nets[0](imgs[mask]).softmax(1)

                s_softmax = s_out.detach().softmax(dim=1)
                all_margin = t_out[better_mask][target[better_mask]] - s_softmax[better_mask][target[better_mask]]
                avg_margin = (all_margin).mean()
                # median_margin = torch.median(all_margin.view(-1))
                s_max_with_margin = s_softmax.max(dim=1).values + avg_margin
                # # s_max_with_margin[s_max_with_margin>1] = s_softmax.max(dim=1).values[s_max_with_margin>1] 
                s_max_with_margin[s_max_with_margin>1] = 1
                s_out_margin_improve = s_softmax.clone()
                s_out_margin_improve *= ((1-s_max_with_margin)/(1-s_softmax.max(dim=1).values))[:,None]
                s_out_margin_improve[s_target] = s_max_with_margin
                # other_sum = 1-t_out[worse_mask][target[worse_mask]]
                # t_out[worse_mask] *= (1-s_max_with_margin[worse_mask])/other_sum
                # t_out[worse_mask][target[worse_mask]] = s_max_with_margin[worse_mask]
                # mask = better_mask

                s_target_margin_improve = soft_target.clone()
                # s_target_margin = s_softmax[target] + avg_margin
                # s_target_margin = s_softmax[target] + median_margin
                s_target_margin = s_softmax[target] + 0.05
                # # mask = mask & (s_target_margin<1)
                s_target_margin[s_target_margin>1] = 1
                # s_target_margin[s_target_margin>1] = 0.9
                s_target_margin_improve[mask] = s_softmax[mask]
                s_target_margin_improve *= ((1-s_target_margin)/(1-s_softmax[target]))[:,None]
                s_target_margin_improve[target] = s_target_margin
                s_target_margin_improve /= s_target_margin_improve.sum(1)[:,None]

                t_target_margin_improve = t_out.clone()
                t_target_margin_improve *= ((1-s_target_margin)/(1-t_out[target]))[:,None]
                t_target_margin_improve[target] = s_target_margin
                t_target_margin_improve /= t_target_margin_improve.sum(1)[:,None]

                # t_out[mask] = s_out_margin_improve[mask]
                # t_out[mask] = soft_target[mask]
                # t_out[mask] = s_target_margin_improve[mask]
                # t_out[mask] = t_target_margin_improve[mask]
                t_out[mask] = target.to(t_out.dtype)[mask]
                # t_out[mask] = t0_mask_out
            #     logs['s_confidence_mask'] = (s_softmax[mask][target[mask]]).mean()
            # else:
            #     logs['s_confidence_mask'] = torch.tensor(0)

            logs['percentage'] = mask.float().mean()
            # logs['s_mask_larger_percent'] = (s_softmax[mask][target[mask]]>t_out[mask][target[mask]]).sum().float()/s_softmax.shape[0]
            # logs['s_mask_larger_percent'] = (s_softmax[mask][target[mask]]>t_out[mask][target[mask]]).float().mean()
            # logs['final_confidence_mask'] = (t_out[mask][target[mask]]).mean()
            # logs['t0_confidence_mask'] = (t0_mask_out[target[mask]]).mean()
            # logs['t1_confidence_mask'] = (t1_out.softmax(dim=1)[mask][target[mask]]).mean()
            # t0_confidence_mask: 0.8983, t1_confidence_mask: 0.7168, s_confidence_mask: 0.8 for worse_mask & (~t_wrong_mask) & (~s_wrong_mask) 
            # t_s_diff = t_s_diff_mask.sum().float() / imgs.shape[0]
        # t_out = torch.zeros_like(s_out)
        # t_out.scatter_(1, labels[:,None], 1)
        # t_out = t_out*0.9 + 0.1/1000

        losses = self.get_loss(s_out, t_out, labels)
        losses.update(logs)
        # losses['t_s_diff'] = t_s_diff
        return losses

