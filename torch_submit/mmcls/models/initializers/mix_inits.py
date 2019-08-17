import math
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import kaiming_init, constant_init

__all__ = ['kaiming_conv_const_bn', 'dw_conv']


def kaiming_conv_const_bn(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m)
        elif isinstance(m, _BatchNorm):
            constant_init(m, 1)


def dw_conv(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            n /= m.groups
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
            m.eps = 1e-3
            m.momentum = 0.1
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                m.bias.data.zero_()
