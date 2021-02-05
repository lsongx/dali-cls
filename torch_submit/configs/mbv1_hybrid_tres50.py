import nvidia.dali.types as types
# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='Hybrid',
    teacher_net=dict(
        type='SequenceResNet',
        depth=50,
        implement='local'),
    student_net=dict(
        type='SequenceMobilenetV1',
        implement='local'),
    # loss=dict(
    #     type='CrossEntropySmoothLoss',
    #     implement='local',
    #     smoothing=0.1), # high-baseline
    loss=dict(type='CrossEntropyLoss'),
    # teacher_connect_index=(7, 15, 51, 54),
    # student_connect_index=(7, 11, 17, 20),
    teacher_connect_index=(11, 20),
    student_connect_index=(8, 18),
    student_channels=(256, 1024),
    teacher_pretrained='./data/resnet50-19c8e357.pth',
    # student_backbone_init_cfg='dw_conv', # high-baseline
    ori_net_path_loss_alpha=0.9)
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
            path=["./data/train_orig.rec"], 
            index_path=["./data/train_orig.idx"])),
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
            path=["./data/val_c224_q95.rec"],
            index_path=["./data/val_c224_q95.idx"])),
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
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=4e-5)
# learning policy
lr_config = dict(policy='cosine', warmup='linear', warmup_iters=5000, target_lr=1e-4, by_epoch=False)
# misc settings
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(interval=1, switch_loader_epoch=250)
# extra_hooks = [
#     dict(
#         type='ModelParamAdjustHook',
#         param_name_adjust_epoch_value = [
#             ('ori_net_path_loss_alpha', 0, 0.2),
#             ('ori_net_path_loss_alpha', 40, 0.6),
#             ('ori_net_path_loss_alpha', 60, 0.9),
#             ('ori_net_path_loss_alpha', 80, 1)],)]
total_epochs = 120
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
