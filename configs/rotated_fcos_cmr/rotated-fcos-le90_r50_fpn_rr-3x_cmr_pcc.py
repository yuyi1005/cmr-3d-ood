_base_ = [
    '../_base_/datasets/cmr.py', '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'r360'

# model settings
model = dict(
    type='mmdet.FCOS',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSCMRHead',
        num_classes=1,
        in_channels=256,
        angle_version=angle_version,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        scale_angle=False,
        angle_coder=dict(
            type='PSC3DCoder',
            angle_version=angle_version,
            omega=1,
            num_step=3),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.IoULoss', loss_weight=2.0),
        loss_angle=dict(
            type='mmdet.L1Loss', loss_weight=2.0),
        use_hbbox_loss=True,
        use_cosine_similarity_loss=True,
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='rbox'),
    dict(type='mmdet.FixShapeResize', width=256, height=256, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=0.5, angle_range=180),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(data_root='data/cmr-3d-ood/acdc', pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(data_root='data/cmr-3d-ood/acdc'))

test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        data_root='data/cmr-3d-ood/acdc',
        ann_file='test/labels/',
        data_prefix=dict(img_path='test/images/')
    )
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=6)

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=6,
        max_keep_ckpts=1
    )
)
