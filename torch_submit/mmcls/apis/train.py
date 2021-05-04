from __future__ import division

import torch
from mmcv.runner import (DistSamplerSeedHook, Fp16OptimizerHook, obj_from_dict,
                         OptimizerHook, build_optimizer, build_runner)
from mmcv.parallel import MMDistributedDataParallel

import mmcls
from mmcls.core import DistOptimizerHook, DistEvalTopKHook
from mmcls.datasets import build_dataloader

from .env import get_root_logger


def train_model(model,
                cfg,
                distributed=False,
                validate=False,
                logger=None,
                meta=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, cfg, validate=validate, logger=logger, meta=meta)
    else:
        _non_dist_train(model, cfg, validate=validate, logger=logger)


def _dist_train(model, cfg, validate=False, logger=None, meta=None):
    # prepare data loaders
    data_loaders = [
        build_dataloader(cfg.data.train_cfg, cfg.local_rank, cfg.world_size)
    ]

    # put model on gpus
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=False)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    optimizer_config = cfg.get('optimizer_config', {})
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **optimizer_config, **fp16_cfg, distributed=True)
    elif 'type' not in optimizer_config:
        optimizer_config = OptimizerHook(**optimizer_config)
    else:
        optimizer_config = optimizer_config

    # register hooks
    lr_config_implement = cfg.lr_config.pop('implement', 'mmcv')
    if lr_config_implement == 'local':
        lr_config = obj_from_dict(cfg.lr_config, mmcls.core.hooks)
    else:
        lr_config = cfg.lr_config
    runner.register_training_hooks(lr_config=lr_config, 
                                   checkpoint_config=cfg.checkpoint_config,
                                   optimizer_config=optimizer_config,
                                   log_config=cfg.log_config)
    # runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_loader_fast = build_dataloader(
            cfg.data.val_cfg_fast, cfg.local_rank, cfg.world_size)
        val_cfg_accurate = cfg.data.get('val_cfg_accurate', None)
        if val_cfg_accurate is not None:
            val_loader_accurate = build_dataloader(
                cfg.data.val_cfg_accurate, cfg.local_rank, cfg.world_size)
        else:
            val_loader_accurate = None
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['logger'] = logger
        runner.register_hook(
            DistEvalTopKHook(val_loader_fast, val_loader_accurate, **eval_cfg))
    for hook in cfg.get('extra_hooks', []):
        hook['logger'] = logger
        runner.register_hook(obj_from_dict(hook, mmcls.core.hooks))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def _non_dist_train(model, cfg, validate=False, logger=None):
    raise NotImplementedError
    # # prepare data loaders
    # data_loaders = [
    #     build_dataloader(
    #         dataset,
    #         cfg.data.imgs_per_gpu,
    #         cfg.data.workers_per_gpu,
    #         cfg.gpus,
    #         dist=False)
    # ]
    # # put model on gpus
    # model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # # build runner
    # optimizer = build_optimizer(model, cfg.optimizer)
    # runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
    #                 cfg.log_level)
    # # fp16 setting
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     optimizer_config = Fp16OptimizerHook(
    #         **cfg.optimizer_config, **fp16_cfg, distributed=False)
    # else:
    #     optimizer_config = cfg.optimizer_config
    # runner.register_training_hooks(cfg.lr_config, optimizer_config,
    #                                cfg.checkpoint_config, cfg.log_config)

    # if cfg.resume_from:
    #     runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    #     runner.load_checkpoint(cfg.load_from)
    # runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
