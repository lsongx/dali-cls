import logging

import mmcv
from mmcv.runner import obj_from_dict
# ------- there is a bug in mmcv 2.10
# from mmcv.runner import load_checkpoint 
from mmcls.utils.checkpoint import load_checkpoint

import numpy as np
import torch.nn as nn

import mmcls
from mmcls.core import auto_fp16, force_fp32
from mmcls.core.evaluation import accuracy

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
            logger = logging.getLogger()
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
