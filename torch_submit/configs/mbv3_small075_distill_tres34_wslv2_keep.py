import nvidia.dali.types as types
# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='Distill',
    teacher_nets=[dict(
        type='resnet34',
        checkpoint_path='./data/resnet34-333f7ec4.pth',
        implement='torchvision'),],
    # teacher_nets=[dict(
    #     type='mobilenetv2_100',
    #     checkpoint_path='./data/mobilenetv2_100_ra-b33bc2c4.pth',
    #     # type='mobilenetv3_large_100',
    #     # checkpoint_path='./data/mobilenetv3_large_100_ra-f55367f5.pth',
    #     implement='timm'),],
    student_net=dict(
        type='tf_mobilenetv3_small_075',
        implement='timm'),
    distill_loss=dict(
        type='WSLLoss',
        with_soft_target=False,
        temperature=2,
        implement='local',),
    ce_loss=dict(
        type='CrossEntropySmoothLoss',
        implement='local',
        smoothing=0.1),
    ce_loss_alpha=1,
    # distill_loss_alpha=1,
    distill_loss_alpha=0.5,
    backbone_init_cfg='dw_conv')

# dataset settings
data = dict(
    train_cfg=dict(
        type='train',
        engine='dali',
        batch_size=256,
        num_threads=16,
        augmentations=[
            dict(type='ImageDecoder', device='mixed'),
            dict(
                type='RandomResizedCrop', 
                device='gpu',
                size=224,
                random_area=[0.08, 1.0],
                min_filter=types.INTERP_TRIANGULAR,
                mag_filter=types.INTERP_LANCZOS3,
                minibatch_size=16),
            dict(
                type='CropMirrorNormalize', 
                device='gpu', 
                crop=(224, 224),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                run_params=[dict(type='CoinFlip', probability=0.5, key='mirror')])],
        reader_cfg=dict(
            type='MXNetReader',
            path=["./data/train_q95.rec"], 
            index_path=["./data/train_q95.idx"])),
    val_cfg_fast=dict(
        type='val',
        engine='dali',
        batch_size=64,
        num_threads=8,
        augmentations=[
            dict(type='ImageDecoder', device='mixed'),
            dict(
                type='Resize', 
                device='gpu',
                resize_shorter=256,
                min_filter=types.INTERP_TRIANGULAR,
                mag_filter=types.INTERP_LANCZOS3,
                minibatch_size=16),
            dict(
                type='CropMirrorNormalize',
                device='gpu',
                crop=(224, 224),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],)],
        reader_cfg=dict(
            type='MXNetReader',
            path=["./data/val_q95.rec"],
            index_path=["./data/val_q95.idx"])),)

# optimizer
optimizer = dict(
    type='SGD', 
    lr=0.5, 
    momentum=0.9, 
    weight_decay=4e-5,
    paramwise_cfg=dict(norm_decay_mult=0))
# learning policy
# lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=1252, min_lr=1e-4, by_epoch=False)
lr_config = dict(
    type='CosineAnnealingLrUpdaterHook', 
    max_progress=240*1251, 
    min_lr=1e-5, 
    by_epoch=False, 
    implement='local')
runner = dict(type='EpochBasedRunner', max_epochs=300)
# misc settings
extra_hooks = [
    dict(
        type='WSLv2Hook',
        switch_epoch=240,
        optimizer_cfg=dict(
            type='SGD', 
            lr=5e-2, 
            momentum=0.9, 
            weight_decay=0),
        lr_config=dict(
            type='CosineAnnealingLrUpdaterHook', 
            max_progress=62*1251, 
            base_progress=239*1251,
            min_lr=5e-4, 
            by_epoch=False,),
        loss=dict(
            type='WSLLoss',
            # temperature=1,
            temperature=0.8,
            only_teacher_temperature=True,
            with_soft_target=False,
            remove_not_noisy_reg=True,
            implement='local',),
        teacher_nets=[dict(
            type='mobilenetv3_large_100',
            checkpoint_path='./data/mobilenetv3_large_100_ra-f55367f5.pth',
            # type='resnet50',
            # checkpoint_path='./data/resnet50_ram-a26f946b.pth',
            implement='timm'),],)]
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(interval=1, switch_loader_epoch=1e5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
