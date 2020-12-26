def accuracy(output, target, topk=(1,), return_mean=True):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.to(pred.dtype).view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        if return_mean:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k)
    return res
