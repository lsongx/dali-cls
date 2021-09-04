import nvidia.dali.types as types
# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='Distill',
    teacher_nets=[
        # dict(
        #     type='gluon_senet154',
        #     checkpoint_path='./data/gluon_senet154-70a1a3c0.pth',
        #     implement='timm'),
        dict(
            type='gluon_resnet152_v1s',
            checkpoint_path='./data/gluon_resnet152_v1s-dcc41b81.pth',
            implement='timm')],
        # dict(
        #     type='mobilenetv3_large_100',
        #     checkpoint_path='./data/mobilenetv3_large_100_ra-f55367f5.pth',
        #     # type='resnet50',
        #     # checkpoint_path='./data/resnet50_ram-a26f946b.pth',
        #     implement='timm'),],
    student_net=dict(
        type='resnet18d',
        checkpoint_path='./data/checkpoint_74.98.pth',
        implement='timm'),
    ce_loss=dict(
        type='CrossEntropySmoothLoss',
        implement='local',
        smoothing=0.1),
    # distill_loss=dict(
    #     type='KLLoss',
    #     with_soft_target=True,
    #     implement='local',),
    distill_loss=dict(
        type='WSLLoss',
        # temperature=0.9,
        temperature=1,
        only_teacher_temperature=True,
        with_soft_target=False,
        remove_not_noisy_reg=True,
        implement='local',),
    ce_loss_alpha=0,
    distill_loss_alpha=1,
    # backbone_init_cfg='dw_conv',
    pretrained=None)

# dataset settings
data = dict(
    train_cfg=dict(
        type='train',
        engine='dali',
        batch_size=128,
        num_threads=4,
        augmentations=[
            dict(type='ImageDecoder', device='mixed'),
            dict(
                type='RandomResizedCrop', 
                device='gpu',
                size=224,
                random_area=[0.08, 1.0],
                min_filter=types.INTERP_TRIANGULAR,
                mag_filter=types.INTERP_LANCZOS3,
                minibatch_size=4),
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
        batch_size=16,
        num_threads=4,
        augmentations=[
            dict(type='ImageDecoder', device='mixed'),
            dict(
                type='Resize', 
                device='gpu',
                resize_shorter=256,
                min_filter=types.INTERP_TRIANGULAR,
                mag_filter=types.INTERP_LANCZOS3,
                minibatch_size=4),
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
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0)
# learning policy
# lr_config = dict(policy='Step', step=[100])
# runner = dict(type='EpochBasedRunner', max_epochs=180)
# lr_config = dict(policy='Step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=80)
lr_config = dict(policy='CosineAnnealing', min_lr=5e-4, by_epoch=False)
# runner = dict(type='EpochBasedRunner', max_epochs=60)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(interval=1, switch_loader_epoch=1e5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
