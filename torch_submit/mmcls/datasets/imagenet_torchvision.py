from functools import partial

from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from .sampler import DistributedSampler


def build_torchvision_loader(cfg, local_rank, world_size):
    cfg_type = cfg.pop('type')
    if cfg_type == 'val':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        dataset = ImageFolder(**cfg.pop('dataset_cfg'),
                              transform=val_transform)
        sampler = DistributedSampler(
            dataset, world_size, local_rank, shuffle=False, round_up=False)
        loader = DataLoader(
            dataset,
            sampler=sampler,
            num_workers=cfg.pop('num_workers'),
            collate_fn=partial(collate, samples_per_gpu=cfg.pop('batch_size')),
            pin_memory=False)
        return loader
    else:
        raise NotImplementedError
