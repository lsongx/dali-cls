import os
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import mmcv
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate

from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from .accuracy import accuracy
from ..utils import save_checkpoint, AverageMeter


class DistEvalHook(Hook):

    def __init__(self,
                 dataloader_fast,
                 dataloader_accurate,
                 switch_loader_epoch=100,
                 interval=1,
                 out_dir=None,
                 logger=None):
        self.dataloader_fast = dataloader_fast
        self.dataloader_accurate = dataloader_accurate
        self.dataloader = self.dataloader_fast
        self.switch_loader_epoch = switch_loader_epoch
        self.interval = interval
        self.out_dir = out_dir
        self.best_top1 = 0
        self.logger = logger

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        if runner.epoch > self.switch_loader_epoch:
            if self.dataloader_accurate is not None:
                self.dataloader = self.dataloader_accurate

        top1 = self.evaluate(runner)

        if runner.rank == 0:
            if not self.out_dir:
                self.out_dir = runner.work_dir
            is_best = False
            if top1 > self.best_top1:
                is_best = True
                old_filename = f'checkpoint_{self.best_top1:.2f}.pth'
                if os.path.isfile(osp.join(self.out_dir, old_filename)):
                    os.remove(osp.join(self.out_dir, old_filename))
                self.best_top1 = top1
                self.bestname = f'checkpoint_{self.best_top1:.2f}.pth'
                if self.logger is not None:
                    self.logger.info(f'Saving best {self.bestname}.')
            save_checkpoint(runner.model.module.get_model(), is_best, self.out_dir,
                            bestname=self.bestname)
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError

    @staticmethod
    def reduce_tensor(tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= torch.distributed.get_world_size()
        return rt


# from ..utils import tensor2imgs
# from PIL import Image

class DistEvalTopKHook(DistEvalHook):

    def evaluate(self, runner):
        runner.model.eval()

        if isinstance(self.dataloader, DALIClassificationIterator):
            top1 = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
            top5 = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
            size = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
            for i, data in enumerate(self.dataloader):
                x = data[0]["data"]
                y = data[0]["label"].squeeze().cuda(non_blocking=True).long()
                with torch.no_grad():
                    output = runner.model(x, y, return_loss=False)
                prec1, prec5 = accuracy(output, y, (1, 5), False)
                top1 += prec1
                top5 += prec5
                size += x.shape[0]
            dist.all_reduce(top1, op=dist.ReduceOp.SUM)
            dist.all_reduce(top5, op=dist.ReduceOp.SUM)
            dist.all_reduce(size, op=dist.ReduceOp.SUM)
            top1 = top1.item()*100/size.item()
            top5 = top5.item()*100/size.item()
            runner.log_buffer.output['Top1'] = top1
            runner.log_buffer.output['Top5'] = top5
            runner.log_buffer.ready = True
            return top1
        elif isinstance(self.dataloader, torch.utils.data.DataLoader):
            top1 = torch.zeros(1, device=f'cuda:{runner.rank}')
            top5 = torch.zeros(1, device=f'cuda:{runner.rank}')
            size = torch.tensor(len(self.dataloader.dataset),
                                device=f'cuda:{runner.rank}')
            # from tqdm import tqdm
            # for i, data in tqdm(enumerate(self.dataloader)):
            for i, data in enumerate(self.dataloader):
                x, y = data
                x = x.to(runner.rank)
                y = y.to(runner.rank)
                with torch.no_grad():
                    output = runner.model(x, y, return_loss=False)
                prec1, prec5 = accuracy(output, y, (1, 5), False)
                top1 += prec1
                top5 += prec5
            dist.all_reduce(top1, op=dist.ReduceOp.SUM)
            dist.all_reduce(top5, op=dist.ReduceOp.SUM)
            top1 = top1.item()*100/size.item()
            top5 = top5.item()*100/size.item()
            runner.log_buffer.output['Top1'] = top1
            runner.log_buffer.output['Top5'] = top5
            runner.log_buffer.ready = True
            return top1
        else:
            raise NotImplementedError(
                f'Not supported type for loader {type(self.dataloader)}')


