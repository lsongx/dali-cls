import mmcls


def build_dataloader(cfg, local_rank, world_size):
    get_loader = getattr(mmcls.datasets, f'build_{cfg.pop("engine")}_loader')
    return get_loader(cfg, local_rank, world_size)
