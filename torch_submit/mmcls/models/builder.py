from torch import nn

import mmcls
from mmcls.utils import build_from_cfg
from .registry import BACKBONES, LOSSES, CLASSIFIERS


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    implement_source = cfg.pop('implement', 'torchvision')
    if implement_source == 'torchvision':
        model = getattr(mmcls.models.backbones, cfg.pop('type'))
        return model(**cfg)
    else:
        return build(cfg, BACKBONES)


def build_loss(cfg):
    implement_source = cfg.pop('implement', 'torch')
    if implement_source == 'torch':
        model = getattr(mmcls.models.loss, cfg.pop('type'))
        return model(**cfg)
    else:
        return build(cfg, LOSSES)


def build_classifier(cfg):
    return build(cfg, CLASSIFIERS)
