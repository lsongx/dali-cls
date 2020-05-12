import logging
import torch

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
class CodingClassifier(nn.Module):

    def __init__(self, 
                 backbone, 
                 loss, 
                 num_classes=1000,
                 code_book='onehot',
                 pretrained=None, 
                 backbone_init_cfg=None):
        super(CodingClassifier, self).__init__()
        self.fp16_enabled = False
        self.backbone = builder.build_backbone(backbone)
        self.loss = builder.build_loss(loss)
        self.init_weights(pretrained, backbone_init_cfg)
        self.init_code_book(num_classes, code_book)

    def init_code_book(self, num_classes, code_book):
        if code_book == 'onehot':
            self.code_book = nn.Parameter(
                torch.diag(torch.ones(num_classes)), requires_grad=False)
        elif code_book == '':
            pass
        else:
            raise NotImplementedError(
                f'Code book class {code_book} not implemented')

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
        losses = self.get_loss(outputs, labels, self.code_book)
        return losses

    def forward_test(self, imgs, labels):
        outputs = self.backbone(imgs)
        outputs = -self.get_hamming_distance(
            outputs[:,None], self.code_book[None])
        return outputs

    @force_fp32(apply_to=('outputs', 'labels', 'code_book'))
    def get_loss(self, outputs, labels, code_book):
        losses = dict()
        losses['loss'] = self.loss(outputs, labels, code_book)
        losses['acc'] = self.accuracy(outputs, labels, code_book)
        return losses

    def get_model(self):
        # return self.backbone.state_dict()
        return self.state_dict()

    def get_hamming_distance(self, output, code):
        """
        Input:
            output: N*None*C
            code: None*1000*C
        Return: N*1000
        """
        return -(output*code + (1-output)*(1-code)).sum(dim=2) / code.shape[1]

    # fp32 for compatibility
    @force_fp32(apply_to=('outputs', 'labels', 'code_book'))
    def accuracy(self, outputs, labels, code_book):
        outputs = self.get_hamming_distance(outputs[:,None], code_book[None])
        return accuracy(-outputs, labels)
