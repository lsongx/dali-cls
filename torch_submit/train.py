import argparse
import time
import os

import mmcv
from mmcv import Config
from mmcv.utils import collect_env

from mmcls.utils import update_cfg_from_args, get_root_logger
from mmcls.apis import train_model, init_dist, set_random_seed
from mmcls.models import build_classifier
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier.')
    # parser.add_argument('--config', default='./configs/res50_ce.py', help='train config file path')
    parser.add_argument('--config', default='./configs/res18_ce.py', help='train config file path')
    parser.add_argument('--work_dir', default='./data/out',
                        help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate', default=1, type=int,
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        '--data.train_cfg.reader_cfg.path', type=str,
        # default='~/data/imagenet/imagenet-rec-save/train_orig.rec'
        default='~/data/imagenet/imagenet-rec-save/train_q95.rec'
    )
    parser.add_argument(
        '--data.train_cfg.reader_cfg.index_path', type=str,
        # default='~/data/imagenet/imagenet-rec-save/train_orig.idx'
        default='~/data/imagenet/imagenet-rec-save/train_q95.idx'
    )
    parser.add_argument(
        '--data.val_cfg_fast.reader_cfg.path', type=str,
        default='~/data/imagenet/imagenet-rec-save/val_q95.rec'
    )
    parser.add_argument(
        '--data.val_cfg_fast.reader_cfg.index_path', type=str,
        default='~/data/imagenet/imagenet-rec-save/val_q95.idx'
    )
    # parser.add_argument(
    #     '--data.val_cfg_fast.reader_cfg.file_root', type=str,
    #     default='~/data/imagenet/val'
    # )
    # parser.add_argument(
    #     '--data.val_cfg_accurate.dataset_cfg.root', type=str,
    #     default='~/data/imagenet/val'
    # )
    # parser.add_argument('--model.pretrained', type=str, 
    #                     # default='~/data/models/resnet50-19c8e357.pth')
    #                     default='~/data/models/resnet18-5c106cde.pth')

    parser.add_argument('--data.train_cfg.batch_size', type=int, default=128)
    parser.add_argument('--log_config.interval', type=int, default=1)
    parser.add_argument('--optimizer.lr', type=float, default=0.1)

    parser.add_argument('--use_fp16', type=int, default=1)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if not args.use_fp16:
        cfg.pop('fp16', {})
    update_cfg_from_args(args, cfg)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    if args.local_rank == 0:
        logger.info(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    meta['env_info'] = env_info

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_classifier(cfg.model)

    cfg.local_rank = args.local_rank
    cfg.world_size = torch.distributed.get_world_size()

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(config=cfg.text)

    train_model(
        model,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
