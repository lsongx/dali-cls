import torch
from ..registry import LOSSES


@LOSSES.register_module
class WSLLoss(torch.nn.Module):
    """ Compute loss given inputs and labels.
    Args:
        inputs: Input tensor of the criterion.
        labels: Label tensor of the criterion.
    """

    def __init__(self, 
                 beta=1, 
                 with_soft_target=True, 
                 temperature=1, 
                 only_teacher_temperature=False,
                 remove_not_noisy_reg=False):
        super().__init__()
        self.beta = beta
        self.with_soft_target = with_soft_target
        self.temperature = temperature
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.nll_loss = torch.nn.NLLLoss(reduction='none')
        self.only_teacher_temperature = only_teacher_temperature
        self.remove_not_noisy_reg = remove_not_noisy_reg

    def forward(self, student, teacher, label):
        if not self.with_soft_target:
            teacher = (teacher/self.temperature).softmax(dim=1)

        if self.only_teacher_temperature:
            student_logsoftmax_with_t = self.logsoftmax(student)
        else:
            student_logsoftmax_with_t = self.logsoftmax(student/self.temperature)
        softmax_loss_s = self.nll_loss(student_logsoftmax_with_t, label)
        softmax_loss_t = self.nll_loss(teacher.log(), label)

        wsl_weight = (softmax_loss_s / softmax_loss_t).detach()
        wsl_weight[wsl_weight<0] = 0
        wsl_weight = 1 - torch.exp(- wsl_weight) 

        if self.remove_not_noisy_reg:
            t_wrong_mask = teacher.max(1).indices!=label
            mask = (softmax_loss_s<softmax_loss_t) & (~t_wrong_mask)
            wsl_weight[mask] = 0
            wsl_weight[~mask] = 1

            target_mask = torch.zeros_like(student).bool()
            target_mask = target_mask.scatter_(1, label[:,None], 1)
            soft_target = target_mask.to(teacher.dtype)*0.99+0.01/1000
            student_logsoftmax = self.logsoftmax(student)
            ce_loss = torch.sum(-soft_target * student_logsoftmax, 1)
            distill_loss = torch.sum(-teacher * student_logsoftmax_with_t, 1)
            distill_loss *= wsl_weight**self.beta
            if not self.only_teacher_temperature:
                distill_loss *= self.temperature**2

            batch_loss = ce_loss*(1-wsl_weight) + distill_loss*wsl_weight
        else:
            batch_loss = torch.sum(-teacher * student_logsoftmax_with_t, 1)
            batch_loss *= wsl_weight**self.beta
            if not self.only_teacher_temperature:
                batch_loss *= self.temperature**2

        return torch.mean(batch_loss)
