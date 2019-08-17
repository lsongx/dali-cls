# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='BaseClassifier',
    backbone=dict(
        type='MobilenetV1',
        implement='local'),
    loss=dict(
        type='CrossEntropySmoothLoss',
        implement='local',
        smoothing=0.1),
    backbone_init_cfg='dw_conv')
# dataset settings
data = dict(
    train_cfg=dict(
        type='train',
        engine='dali',
        batch_size=128,
        num_threads=16,
        augmentations=[
            dict(type='ImageDecoderRandomCrop', device='mixed'),
            dict(type='Resize', device='gpu', resize_x=224, resize_y=224),
            dict(
                type='ColorTwist',
                device='gpu',
                run_params=[
                    dict(type='Uniform', range=[0.6, 1.4], key='brightness'),
                    dict(type='Uniform', range=[0.6, 1.4], key='contrast'),
                    dict(type='Uniform', range=[0.6, 1.4], key='saturation'),
                ]),
            dict(
                type='CropMirrorNormalize',
                device='gpu',
                crop=224,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                run_params=[
                    dict(type='CoinFlip', probability=0.5, key='mirror')
                ])
        ],
        reader_cfg=dict(
            type='MXNetReader',
            path=["./data/train_orig.rec"], 
            index_path=["./data/train_orig.idx"])),
    val_cfg_fast=dict(
        type='val',
        engine='dali',
        batch_size=64,
        num_threads=8,
        reader_cfg=dict(
            type='MXNetReader',
            path=["./data/train_orig.rec"],
            index_path=["./data/train_orig.idx"])),
        # reader_cfg=dict(
        #     type='FileReader',
        #     file_root=["./data/val"])))
    val_cfg_accurate=dict(
        type='val',
        engine='torchvision',
        batch_size=64,
        num_workers=8,
        dataset_cfg=dict(root="./data/val")))
# optimizer
optimizer = dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=4e-5)
# learning policy
lr_config = dict(policy='cosine', warmup='linear', warmup_iters=1252, target_lr=1e-4, by_epoch=False)
# misc settings
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(interval=1, switch_loader_epoch=100)
total_epochs = 241
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
