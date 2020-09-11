import math
import torch

from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from mmcv.runner import obj_from_dict


def _init_run_params(run_params):
    """return a tuple (calling_method, input_param_dict_key, whether_to_call)
    """
    if run_params is None:
        return None, None, False
    for param in run_params:
        input_param_dict_key = param.pop('key')
        calling_method = obj_from_dict(param, ops)
        return (calling_method, input_param_dict_key, True)


def _init_augmentations(augmentations):
    aug_list = []
    calling_method_list = []
    input_param_dict_key_list = []
    whether_to_call_list = []
    for aug in augmentations:
        run_params = aug.pop('run_params', None)
        calling, dict_key, wtc = _init_run_params(run_params)
        calling_method_list.append(calling)
        input_param_dict_key_list.append(dict_key)
        whether_to_call_list.append(wtc)
        aug_list.append(obj_from_dict(aug, ops))
    return (aug_list, calling_method_list, 
            input_param_dict_key_list, whether_to_call_list)


class DALITrainPipe(Pipeline):
    def __init__(self, 
                 local_rank, 
                 world_size,
                 batch_size, 
                 num_threads, 
                 reader_cfg,
                 augmentations):
        super(DALITrainPipe, self).__init__(batch_size, num_threads, 
                                            local_rank, seed=12 + local_rank)
        reader = getattr(ops, reader_cfg.pop('type'))
        self.input = reader(shard_id=local_rank,
                            num_shards=world_size,
                            random_shuffle=True,
                            # read_ahead=True,
                            **reader_cfg)
        self.aug_list, self.calling_method_list, \
        self.input_param_dict_key_list, self.whether_to_call_list =\
            _init_augmentations(augmentations)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        for aug, call, in_dict_key, wtc in zip(
            self.aug_list, self.calling_method_list,
            self.input_param_dict_key_list, self.whether_to_call_list):
            if wtc:
                tmp_value = call()
                in_dict = {in_dict_key: tmp_value}
                images = aug(images, **in_dict)
            else:
                images = aug(images)
        return [images, labels]


class DALIValPipe(Pipeline):
    def __init__(self, 
                 local_rank,
                 world_size,
                 batch_size, 
                 num_threads, 
                 reader_cfg,
                 augmentations):
        super(DALIValPipe, self).__init__(batch_size, num_threads, 
                                          local_rank, seed=12 + local_rank)
        reader = getattr(ops, reader_cfg.pop('type'))
        self.input = reader(shard_id=local_rank, 
                            num_shards=world_size, 
                            random_shuffle=False, 
                            **reader_cfg)
        self.aug_list, self.calling_method_list, \
        self.input_param_dict_key_list, self.whether_to_call_list =\
            _init_augmentations(augmentations)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images, labels = self.input(name="Reader")
        for aug, call, in_dict_key, wtc in zip(
            self.aug_list, self.calling_method_list,
            self.input_param_dict_key_list, self.whether_to_call_list):
            if wtc:
                tmp_value = call()
                in_dict = {in_dict_key: tmp_value}
                images = aug(images, **in_dict)
            else:
                images = aug(images)
        return [images, labels]


class WarpDALIClassificationIterator(DALIClassificationIterator):
    def __len__(self):
        return self._epoch_length


def build_dali_loader(cfg, local_rank, world_size):
    cfg_type = cfg.pop('type')
    if cfg_type == 'train':
        pipe = DALITrainPipe(local_rank=local_rank, 
                             world_size=world_size, **cfg)
        pipe.build()
        size = int(pipe.epoch_size("Reader") / world_size)
        loader = WarpDALIClassificationIterator(
            pipe, size=size, fill_last_batch=False, auto_reset=True)
        loader._epoch_length = math.ceil(loader._size / cfg.get('batch_size'))
        return loader
    elif cfg_type == 'val':
        pipe = DALIValPipe(local_rank=local_rank, 
                           world_size=world_size, **cfg)
        pipe.build()
        size = int(pipe.epoch_size("Reader") / world_size)
        loader = WarpDALIClassificationIterator(
            pipe, size=size, fill_last_batch=False, auto_reset=True)
        loader._epoch_length = math.ceil(loader._size / cfg.get('batch_size'))
        return loader
    else:
        raise NotImplementedError
