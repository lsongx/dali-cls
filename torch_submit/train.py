import argparse
import os
from mmcv import Config

from mmcls.utils import update_cfg_from_args
from mmcls.apis import train_model, init_dist, get_root_logger, set_random_seed
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
    parser.add_argument('--seed', type=int, default=None, help='random seed')
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
    parser.add_argument(
        '--data.val_cfg_accurate.dataset_cfg.root', type=str,
        default='~/data/imagenet/val'
    )
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
    if args.local_rank == 0:
        print(cfg)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_classifier(cfg.model)

    cfg.local_rank = args.local_rank
    cfg.world_size = torch.distributed.get_world_size()

    train_model(
        model,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
