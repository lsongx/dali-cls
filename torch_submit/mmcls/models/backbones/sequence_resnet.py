from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from ..registry import BACKBONES


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


@BACKBONES.register_module
class SequenceResNet(nn.Module):
    """Warp all layers into nn.sequential to support subscription.
    """
    def __init__(self, depth):
        super(SequenceResNet, self).__init__()
        depth_block_layer_dict = {
            18: (BasicBlock, [2, 2, 2, 2]),
            34: (BasicBlock, [3, 4, 6, 3]),
            50: (Bottleneck, [3, 4, 6, 3]),
            101: (Bottleneck, [3, 4, 23, 3]),
            152: (Bottleneck, [3, 8, 36, 3]),
        }
        self.ori_net = ResNet(*depth_block_layer_dict[depth])
        self.sequence_warp = nn.Sequential(
            self.ori_net.conv1,
            self.ori_net.bn1,
            self.ori_net.relu,
            self.ori_net.maxpool,
            *self.ori_net.layer1.children(),
            *self.ori_net.layer2.children(),
            *self.ori_net.layer3.children(),
            *self.ori_net.layer4.children(),
            self.ori_net.avgpool,
            Flatten(),
            self.ori_net.fc,
        )

    def forward(self, x):
        return self.sequence_warp(x)
