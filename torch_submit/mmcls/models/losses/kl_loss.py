import torch
from ..registry import LOSSES


@LOSSES.register_module
class KLLoss(torch.nn.Module):
    """ Compute loss given inputs and labels.
    Args:
        inputs: Input tensor of the criterion.
        labels: Label tensor of the criterion.
    """

    def __init__(self, smoothing=0.0, with_soft_target=False):
        super().__init__()
        self.label_smoothing = smoothing
        self.with_soft_target = with_soft_target
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, labels):
        if self.with_soft_target:
            return torch.mean(torch.sum(-labels * self.logsoftmax(inputs), 1))
        n_classes = inputs.size(1)
        # convert to one-hot
        target = torch.unsqueeze(labels, 1)
        soft_target = torch.zeros_like(inputs)
        soft_target.scatter_(1, target, 1)
        # label smoothing
        soft_target = soft_target * \
            (1 - self.label_smoothing) + self.label_smoothing / n_classes
        return torch.mean(torch.sum(-soft_target * self.logsoftmax(inputs), 1))
