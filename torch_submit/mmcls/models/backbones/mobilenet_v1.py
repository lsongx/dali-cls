import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import BACKBONES


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


@BACKBONES.register_module
class MobilenetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw_no_relu(128, 128, 1),    # layer 1 out
        )
        self.layer2 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw_no_relu(256, 256, 1),    # layer 2 out
        )
        self.layer3 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw_no_relu(512, 512, 1),    # layer 3 out
        )
        self.layer4 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw_no_relu(1024, 1024, 1),    # layer 4 out
        )
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, 1000)
        self.channels_by_layer = [128, 256, 512, 1024]

    def forward(self, x):
        layer1_feat = self.layer1(x)
        x = self.relu(layer1_feat)
        layer2_feat = self.layer2(x)
        x = self.relu(layer2_feat)
        layer3_feat = self.layer3(x)
        x = self.relu(layer3_feat)
        layer4_feat = self.layer4(x)
        x = self.relu(layer4_feat)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


@BACKBONES.register_module
class MobilenetV1LayerOut(MobilenetV1):

    def forward(self, x):
        layer1_feat = self.layer1(x)
        x = self.relu(layer1_feat)
        layer2_feat = self.layer2(x)
        x = self.relu(layer2_feat)
        layer3_feat = self.layer3(x)
        x = self.relu(layer3_feat)
        layer4_feat = self.layer4(x)
        x = self.relu(layer4_feat)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x, layer1_feat, layer2_feat, layer3_feat, layer4_feat


@BACKBONES.register_module
class MaskMobilenetV1LayerOut(MobilenetV1LayerOut):

    def __init__(self):
        super().__init__()
        self.mask = nn.ParameterList(
            nn.Parameter(i) for i in
            [
                torch.ones([1, self.channels_by_layer[0], 1, 1]),
                torch.ones([1, self.channels_by_layer[1], 1, 1]),
                torch.ones([1, self.channels_by_layer[2], 1, 1]),
                torch.ones([1, self.channels_by_layer[3], 1, 1]),
            ]
        )

    def forward(self, x, use_mask=False):
        layer1_feat = self.layer1(x)
        x = layer1_feat
        if use_mask:
            x = x * self.mask[0]
        x = self.relu(x)

        layer2_feat = self.layer2(x)
        x = layer2_feat
        if use_mask:
            x = x * self.mask[1]
        x = self.relu(x)

        layer3_feat = self.layer3(x)
        x = layer3_feat
        if use_mask:
            x = x * self.mask[2]
        x = self.relu(x)

        layer4_feat = self.layer4(x)
        x = layer4_feat
        if use_mask:
            x = x * self.mask[3]
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x, layer1_feat, layer2_feat, layer3_feat, layer4_feat
