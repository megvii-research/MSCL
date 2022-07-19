_base_ = [
    '../../_base_/models/tsm_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]

# runtime settings
work_dir = './work_dirs/clf_train/tsm_r50_1x1x8_50e_sthv2_rgb/'
default_dir='./work_dirs/clf_train/tsm_r50_1x1x8_50e_sthv2_rgb/'
# model settings
model = dict(cls_head=dict(num_classes=174))

# dataset settings
dataset_type = 'RedisRawframeDataset'
redis_url="redis://redis.wuqian.ws2.hh-b.brainpp.cn:6379/1"
pkl_path = 's3://activity-public/something-something/annos/somethingv2/somethingv2_train.pkl'
pkl_path_val = 's3://activity-public/something-something/annos/somethingv2/somethingv2_val.pkl'
visual_cfg = dict(cur_path=work_dir, default_path=default_dir, dataset_name="sthv2",
                  vis_acc=True, vis_cf=False, k=20)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='NoriDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='NoriDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='NoriDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        redis_url=redis_url,
        pkl_path=pkl_path,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        redis_url=redis_url,
        pkl_path=pkl_path_val,
        pipeline=val_pipeline,
        visual_cfg=visual_cfg),
    test=dict(
        type=dataset_type,
        redis_url=redis_url,
        pkl_path=pkl_path_val,
        pipeline=test_pipeline,
        visual_cfg=visual_cfg))
evaluation = dict(
    interval=2, metrics=['top_k_accuracy', 'vis_mean_class_accuracy'])

# optimizer
optimizer = dict(
    lr=0.0075,  # this lr is used for 8 gpus
    weight_decay=0.0005)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir="/data/mmaction2/tf_logs"),
    ])

