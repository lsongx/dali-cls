import torch
import os
import shutil

from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip


def tensor2imgs(tensor, 
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225), 
                to_bgr=False):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(img, mean, std, to_bgr=to_bgr)
        if img.max() <= 1:
            img = (img*255).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def save_checkpoint(state, 
                    is_best, 
                    out_dir, 
                    filename='checkpoint.pth', 
                    bestname='model_best.pth'):
    save_dir = os.path.join(out_dir, filename)
    save_dir_best = os.path.join(out_dir, bestname)
    torch.save(state, save_dir)
    if is_best:
        shutil.copyfile(save_dir, save_dir_best)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
