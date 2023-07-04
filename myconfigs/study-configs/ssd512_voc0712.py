input_size = 512
model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[1, 1, 1],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='SSDVGG',
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://vgg16_caffe')),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256, 256),
        level_strides=(2, 2, 2, 2, 1),
        level_paddings=(1, 1, 1, 1, 1),
        l2_norm_scale=20,
        last_kernel_size=4),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        num_classes=20,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=512,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=([2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2])),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        sampler=dict(type='PseudoSampler'),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
backend_args = None

train_dataloader = dict(
    batch_size=8,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='ConcatDataset',
            ignore_keys=['dataset_type'],
            datasets=[
                dict(
                    type='VOCDataset',
                    data_root='data/VOCdevkit/',
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='Expand',
                            mean=[123.675, 116.28, 103.53],
                            to_rgb=True,
                            ratio_range=(1, 4)),
                        dict(
                            type='MinIoURandomCrop',
                            min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                            min_crop_size=0.3),
                        dict(
                            type='Resize', scale=(512, 512), keep_ratio=False),
                        dict(type='RandomFlip', prob=0.5),
                        dict(
                            type='PhotoMetricDistortion',
                            brightness_delta=32,
                            contrast_range=(0.5, 1.5),
                            saturation_range=(0.5, 1.5),
                            hue_delta=18),
                        dict(type='PackDetInputs')
                    ]),
                dict(
                    type='VOCDataset',
                    data_root='data/VOCdevkit/',
                    ann_file='VOC2012/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='Expand',
                            mean=[123.675, 116.28, 103.53],
                            to_rgb=True,
                            ratio_range=(1, 4)),
                        dict(
                            type='MinIoURandomCrop',
                            min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                            min_crop_size=0.3),
                        dict(
                            type='Resize', scale=(512, 512), keep_ratio=False),
                        dict(type='RandomFlip', prob=0.5),
                        dict(
                            type='PhotoMetricDistortion',
                            brightness_delta=32,
                            contrast_range=(0.5, 1.5),
                            saturation_range=(0.5, 1.5),
                            hue_delta=18),
                        dict(type='PackDetInputs')
                    ])
            ])))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VOCDataset',
        data_root='data/VOCdevkit/',
        ann_file='VOC2007test/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007test/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VOCDataset',
        data_root='data/VOCdevkit/',
        ann_file='VOC2007test/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007test/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 20],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005))
auto_scale_lr = dict(enable=False, base_batch_size=64)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
