import math
import torch
from ..registry import LOSSES


@LOSSES.register_module
class WSLv2Loss(torch.nn.Module):
    """ Compute loss given inputs and labels.
    Args:
        inputs: Input tensor of the criterion.
        labels: Label tensor of the criterion.
    """

    def __init__(self, 
                 remove_not_noisy_reg=False, 
                 use_ce=True,
                 with_soft_target=False, 
                 temperature=1):
        super().__init__()
        self.remove_not_noisy_reg = remove_not_noisy_reg
        self.use_ce = use_ce
        self.temperature = temperature
        self.with_soft_target = with_soft_target
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.nll_loss = torch.nn.NLLLoss(reduction='none')

    def forward(self, student, teacher, label):
        logs = {}
        target_mask = torch.zeros_like(student).bool()
        target_mask = target_mask.scatter_(1, label[:,None], 1)
        soft_target = target_mask.to(teacher.dtype)*0.99+0.01/1000

        student_softmax_with_t = (student/self.temperature).softmax(dim=1)
        student_logsoftmax_with_t = self.logsoftmax(student/self.temperature)
        student_logsoftmax = self.logsoftmax(student)
        student_softmax = student.softmax(dim=1)
        teacher_softmax_with_t = (teacher/self.temperature).softmax(dim=1)

        bias = (student_softmax[target_mask] - 1).abs()
        ts_diff = student_softmax_with_t - teacher_softmax_with_t
        var = (self.temperature*ts_diff[target_mask].abs() - bias).abs()
        weight = 1-(-bias/var).exp().detach()

        # ce_loss = self.nll_loss(student_logsoftmax, label)
        ce_loss = torch.sum(-soft_target * student_logsoftmax, 1)
        kd_loss = torch.sum(-teacher_softmax_with_t * student_logsoftmax_with_t, 1)
        kd_loss *= self.temperature**2

        logs['weight before mask'] = weight.mean()
        if self.remove_not_noisy_reg:
            t_wrong_mask = teacher.max(1).indices!=label
            # s_wrong_mask = student.max(1).indices!=label
            # mask = (ts_diff[target_mask]>0) & (~t_wrong_mask) & (~s_wrong_mask) 
            # mask = (ts_diff[target_mask]>0) & (~t_wrong_mask)
            mask = (var>bias) & (~t_wrong_mask)
            weight[mask] = 0
            weight[~mask] = 1

        if self.use_ce:
            batch_loss = ce_loss*(1-weight) + kd_loss*weight
        else:
            batch_loss = kd_loss*weight
        # batch_loss = ce_loss

        logs['weight'] = weight.mean()
        logs['b>v'] = (bias/var>1).float().mean()
        logs['b<=v'] = (bias/var<=1).float().mean()
        logs['ts_diff'] = ts_diff[target_mask].mean()
        logs['bias'] = bias.mean()
        return {'loss': torch.mean(batch_loss), 'logs': logs}
