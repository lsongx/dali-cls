import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.search_config import *
from operations import *
from tools import utils
from tools.multadds_count import comp_multadds


class Block(nn.Module):
    def __init__(self, in_ch, block_ch, head_op, stack_ops, stride):
        super(Block, self).__init__()
        self.head_layer = OPS[head_op](
            in_ch, block_ch, stride, affine=True, track_running_stats=True)

        modules = []
        for stack_op in stack_ops:
            modules.append(OPS[stack_op](block_ch, block_ch,
                                         1, affine=True, track_running_stats=True))
        self.stack_layers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.head_layer(x)
        x = self.stack_layers(x)
        return x


class Conv1_1_Block(nn.Module):
    def __init__(self, in_ch, block_ch):
        super(Conv1_1_Block, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=block_ch,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(block_ch),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv1_1(x)


class AuxiliaryHead(nn.Module):
    def __init__(self, C, num_classes, dataset):
        """assuming input size 8x8"""
        super(AuxiliaryHead, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU6(inplace=True),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Network(nn.Module):

    def __init__(self, 
                 net_config, 
                 dataset='imagenet', 
                 init_mode='he_fout', 
                 last_dim=1280, 
                 aux_config=[False, None, None, None], 
                 config=None):
        """
        aux_config=[True/False, ch, block_idx, aux_weight]
        net_config=[[in_ch, out_ch], head_op, [stack_ops], num_stack_layers, stride]
        """
        super(Network, self).__init__()
        self.config = config
        self.net_config = self.parse_net_config(net_config)

        self._C_input = self.net_config[0][0][0]

        self._dataset = dataset
        self._num_classes = 10 if self._dataset == 'cifar10' else 1000

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self._C_input, kernel_size=3,
                      stride=1 if self._dataset == 'cifar10' else 2,
                      padding=1, bias=False),
            nn.BatchNorm2d(self._C_input),
            nn.ReLU6(inplace=True)
        )

        # self.head_block = OPS['mbconv_k3_t1'](self._C_input, self._head_dim, 1, True, True)

        self.blocks = nn.ModuleList()
        for config in self.net_config:
            self.blocks.append(Block(config[0][0], config[0][1],
                                     config[1], config[2], config[-1]))

        self.aux_config = aux_config
        if aux_config[0]:
            self.aux_classifier = AuxiliaryHead(aux_config[1],
                                                self._num_classes,
                                                self._dataset)

        self.conv1_1_block = Conv1_1_Block(self.net_config[-1][0][1], last_dim)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_dim, self._num_classes)

        self.init_model(model_init=init_mode)
        self.set_bn_param(self.config.optim.bn_momentum,
                          self.config.optim.bn_eps)

    def forward(self, x):

        block_data = self.input_block(x)

        for i, block in enumerate(self.blocks):
            block_data = block(block_data)
            if self.aux_config[0] and i == self.aux_config[2] and self.training:
                logits_aux = self.aux_classifier(block_data)

        block_data = self.conv1_1_block(block_data)

        out = self.global_pooling(block_data)
        logits = self.classifier(out.view(out.size(0), -1))

        if self.aux_config[0] and self.training:
            return logits, logits_aux
        else:
            return logits

    def init_model(self, model_init='he_fout', init_div_groups=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return

    def parse_net_config(self, net_config):
        str_configs = net_config.split('|')
        return [eval(str_config) for str_config in str_configs]


if __name__ == "__main__":
    from configs.cifar_train_config import net_config

    batch_size = 1
    # dataset = 'cifar10'
    dataset = 'imagenet'

    num_class = 10 if dataset == 'cifar10' else 1000
    img_size = 32 if dataset == 'cifar10' else 224
    input_size = (batch_size, 3, img_size, img_size)

    # input_data = torch.randn(1,3,224,224)
    # label = torch.empty(1, dtype=torch.long).random_(1000)
    # net = Network(32, 'imagenet')

    input_data = torch.randn(input_size)
    label = torch.empty(batch_size, dtype=torch.long).random_(num_class)
    net = Network(net_config, dataset)

    input_data, label = input_data.cuda(), label.cuda()
    net = net.cuda()

    print('\n'.join(map(str, net.net_config)))
    print("Params = %.2fMB" % (utils.count_parameters_in_MB(net)))
    print("Mult Adds = %.2fMB" % comp_multadds(net, input_size=input_size))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    logits = net(input_data)

    loss = criterion(logits, label)
    loss.backward()
