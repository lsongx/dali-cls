import torch
from ..registry import LOSSES


@LOSSES.register_module
class CrossEntropySemanticLoss(torch.nn.Module):
    """ Compute loss given inputs and labels.
    Args:
        inputs: Input tensor of the criterion.
        labels: Label tensor of the criterion.
    """

    def __init__(self, label_mat, smoothing=0.0):
        super().__init__()
        label_mat = torch.load(label_mat, 'cpu')['all']
        self.label_mat = torch.nn.Parameter(label_mat, requires_grad=False)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, labels):
        soft_target = self.label_mat[labels, :]
        return torch.mean(torch.sum(-soft_target * self.logsoftmax(inputs), 1))
