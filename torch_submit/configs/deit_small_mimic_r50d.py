import nvidia.dali.types as types
# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='MimicTransformer',
    teacher_net=dict(
        type='resnet50',
        checkpoint_path='./data/resnet50_ram-a26f946b.pth',
        implement='timm'),
    student_net=dict(
        type='vit_deit_small_patch16_224',
        # drop_rate=0.1,
        # attn_drop_rate=0.1,
        # drop_path_rate=0.3,
        implement='timm'),
    loss=dict(
        type='CrossEntropySmoothLoss',
        implement='local',
        smoothing=0.1), # high-baseline
    teacher_connect_index=7,
    student_connect_index=10,
    # student_connect_index=[9,10,11],
    teacher_channels=1024,
    student_channels=384,
    ori_net_path_loss_alpha=50)

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
            # dict(
            #     type='ColorTwist', 
            #     device='gpu',
            #     run_params=[
            #         dict(type='Uniform', range=[0.6, 1.4], key='brightness'),
            #         dict(type='Uniform', range=[0.6, 1.4], key='contrast'),
            #         dict(type='Uniform', range=[0.6, 1.4], key='saturation'),]),
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
optimizer = dict(type='AdamW', lr=5e-4)
# optimizer = dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=4e-5)
# optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing', 
    warmup='linear', 
    warmup_iters=2, 
    warmup_by_epoch=True, 
    min_lr=1e-5, 
    by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=240)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
