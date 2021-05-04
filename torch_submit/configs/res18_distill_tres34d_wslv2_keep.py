import nvidia.dali.types as types
# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='Distill',
    teacher_nets=[dict(
        type='resnet34d',
        checkpoint_path='./data/resnet34d_ra2-f8dcfcaf.pth',
        implement='timm'),],
    student_net=dict(
        type='resnet18',
        implement='torchvision'),
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
        batch_size=128,
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
                minibatch_size=8),
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
        batch_size=32,
        num_threads=8,
        augmentations=[
            dict(type='ImageDecoder', device='mixed'),
            dict(
                type='Resize', 
                device='gpu',
                resize_shorter=256,
                min_filter=types.INTERP_TRIANGULAR,
                mag_filter=types.INTERP_LANCZOS3,
                minibatch_size=8),
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
optimizer = dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=4e-5)
# optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
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
        optimizer_cfg=dict(type='SGD', lr=1e-1, momentum=0.9, weight_decay=0),
        lr_config=dict(type='CosineAnnealingLrUpdaterHook', 
                       max_progress=60*1251, 
                       base_progress=238*1251,
                    #    min_lr=1e-3, 
                       min_lr=1e-4, 
                       by_epoch=False,),
        loss=dict(type='WSLLoss',
                #   temperature=2,
                  temperature=1,
                  with_soft_target=False,
                  remove_not_noisy_reg=True,
                  implement='local',),)]
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(interval=1, switch_loader_epoch=250)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
