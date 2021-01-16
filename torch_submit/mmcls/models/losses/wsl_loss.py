import torch
from ..registry import LOSSES


@LOSSES.register_module
class WSLLoss(torch.nn.Module):
    """ Compute loss given inputs and labels.
    Args:
        inputs: Input tensor of the criterion.
        labels: Label tensor of the criterion.
    """

    def __init__(self, beta=1, with_soft_target=True):
        super().__init__()
        self.beta = beta
        self.with_soft_target = with_soft_target
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.nll_loss = torch.nn.NLLLoss(reduction='none')

    def forward(self, student, teacher, label):
        if not self.with_soft_target:
            teacher = teacher.softmax(dim=1)

        student_logsoftmax = self.logsoftmax(student)
        softmax_loss_s = self.nll_loss(student_logsoftmax, label)
        softmax_loss_t = self.nll_loss(teacher.log(), label)

        wsl_weight = softmax_loss_s / softmax_loss_t
        wsl_weight[wsl_weight<0] = 0
        wsl_weight = 1 - torch.exp(- wsl_weight) 

        batch_loss = torch.sum(-teacher * student_logsoftmax, 1)
        batch_loss *= wsl_weight.detach()**self.beta

        return torch.mean(batch_loss)
        