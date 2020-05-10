import torch
from ..registry import LOSSES


@LOSSES.register_module
class CodingLoss(torch.nn.Module):
    """ Compute loss given inputs and labels.
    Args:
        inputs: Input tensor of the criterion.
        labels: Label tensor of the criterion.
    """

    def __init__(self):
        super().__init__()

    def get_hamming_distance(self, output, code):
        # output: N*None*C
        # code: None*1000*C
        # return -(output*code + (1-output)*(1-code)).sum(dim=2) / code.shape[1]
        return -(output*code + (1-output)*(1-code)).sum(dim=2)

    def forward(self, inputs, labels, code_book):
        # convert to one-hot
        n_classes = inputs.size(1)
        target = torch.unsqueeze(labels, 1)
        y = torch.zeros_like(inputs)
        y.scatter_(1, target, 1)

        distance = self.get_hamming_distance(inputs[:, None], code_book[None])
        min_dist = distance.min().detach()
        distance = distance - min_dist
        predicted = distance.argmin(dim=1)
        label_distance = distance[y.bool()][:]
        non_label_distance = -distance[~y.bool()].view([y.shape[0], -1])
        loss = 1+ non_label_distance.exp().sum(dim=1) / (-label_distance).exp()

        return loss.log().mean()