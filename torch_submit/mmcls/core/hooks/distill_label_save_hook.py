import os
import shutil
import torch
import torch.distributed as dist

import mmcv
from mmcv.runner import Hook, get_dist_info


class DistillLabelSaveHook(Hook):
    """Adjust the param of a model (by epochs).
    """

    def __init__(self, save_name, save_epoch_gap, logger=None):
        self.save_name = save_name
        self.save_epoch_gap = save_epoch_gap
        self.logger = logger

    def after_train_epoch(self, runner):
        if runner.rank == 0:
            epoch = runner.epoch
            current_save_name = f'{self.save_name}-to-epoch{epoch+1:03d}.pth'
            current_path = os.path.join(runner.work_dir, current_save_name)
            previous_stat_dict = {}
            if (epoch+1) % self.save_epoch_gap != 0:
                before_save_name = f'{self.save_name}-to-epoch{epoch:03d}.pth'
                before_path = os.path.join(runner.work_dir, before_save_name)
                if os.path.isfile(before_path):
                    previous_stat_dict = torch.load(before_path)
                    os.remove(before_path)
                else:
                    self.logger.warning(
                        f'no previous saved dict found for {before_path}')

        tmp_dir = os.path.join(runner.work_dir, 'tmp')
        all_current_epoch = self.collect_cpu(
            tmp_dir, runner.model.module.epoch_teacher_outputs)

        if runner.rank == 0:
            all_current_epoch.update(previous_stat_dict)
            torch.save(all_current_epoch, current_path)
            self.logger.info(f'{current_path} saved')

    def collect_cpu(self, tmp_dir, dict_for_save):
        rank, world_size = get_dist_info()
        mmcv.mkdir_or_exist(tmp_dir)
        torch.save(dict_for_save, os.path.join(tmp_dir, f'part_{rank}.pth'))
        dist.barrier()
        if rank != 0:
            return None
        else:
            # load results of all parts from tmp dir
            part_dict = {}
            for i in range(world_size):
                part_file = os.path.join(tmp_dir, f'part_{i}.pth')
                part_dict.update(torch.load(part_file))
            # remove tmp dir
            shutil.rmtree(tmp_dir)
            return part_dict
