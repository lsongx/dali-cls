from torchvision.models.resnet import ResNet


class SequenceResNet(ResNet):
    """Warp all layers into nn.sequential to support subscription.
    """