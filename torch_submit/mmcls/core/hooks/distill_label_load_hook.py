import os
from mmcv.runner import Hook


class DistillLabelLoadHook(Hook):
    """Adjust the param of a model (by epochs).
    """

    def __init__(self, load_path, logger=None):
        self.logger = logger

        start = 0
        self.epoch_to_save_names = {}
        for file in sorted(os.listdir(load_path)):
            if 'to-epoch' in file:
                end = int(file.split('.pth')[0].split('epoch')[-1])
                for i in range(start, end):
                    self.epoch_to_save_names[i] = file
                start = end
        self.logger.info(f'total {len(self.epoch_to_save_names)} label files')

    def before_train_epoch(self, runner):
        label_file = self.epoch_to_save_names(runner.epoch)
        runner.model.module.save_teacher_outputs = torch.load(label_file)
        self.logger.info(f'total {len(self.epoch_to_save_names)} label files')

