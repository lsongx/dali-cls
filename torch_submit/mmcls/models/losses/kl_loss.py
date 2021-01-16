import torch
from ..registry import LOSSES


@LOSSES.register_module
class KLLoss(torch.nn.Module):
    """ Compute loss given inputs and labels.
    Args:
        inputs: Input tensor of the criterion.
        labels: Label tensor of the criterion.
    """

    def __init__(self, with_soft_target=False, temperature=1):
        super().__init__()
        self.with_soft_target = with_soft_target
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.temperature = temperature

    def forward(self, inputs, labels, *args, **kwargs):
        if self.with_soft_target:
            return torch.mean(torch.sum(-labels * self.logsoftmax(inputs), 1))
        inputs = self.logsoftmax(inputs/self.temperature)
        labels = (labels/self.temperature).softmax(dim=1)
        return torch.mean(torch.sum(-labels*inputs, 1)*(self.temperature**2))
