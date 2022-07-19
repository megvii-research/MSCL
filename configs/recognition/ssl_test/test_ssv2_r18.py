_base_ = [
    '../../_base_/default_runtime.py'
]

# Set pretrained/work_dir/log_dir
# runtime settings
work_dir="./work_dirs/ssl_test/sthv2/modist_coaug_samekn_posp_rm_mx_full1_lr2e-2_ssv2"
default_dir="./work_dirs/ssl_test/sthv2/moco_consistent_aug_lr3e-2_ssv2"
# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='torchvision.r3d_18',
        # pretrained='torchvision://resnet50', # set from work_dir
        ),
    cls_head=dict(
        type='I3DHead',
        num_classes=174,
        in_channels=512,
        spatial_type='none',    # This for torchvision.r3d_18
        dropout_ratio=0.5,),
    # model training and testing settings
    # set ssl_pretrain will inhibit original init_weights()
    train_cfg=dict(ssl_pretrain=dict(
        pretrained = dict(filename="./work_dirs/ssl_train/modist_coaug_samekn_posp_rm_mx_full1_lr2e-2/epoch_400.pth"),
        backbone=dict(prefix='recognizer.encoder_q')
    )),
    test_cfg=dict(average_clips='prob'))
# changeable parameters
image_shape=(112, 112)
crop_shape=128
image_short=128
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
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='NoriDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type="RandomResizedCrop"),
    dict(type='Resize', scale=image_shape, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="Seg2T"),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='NoriDecode'),
    dict(type='Resize', scale=(-1, image_short)),
    dict(type='CenterCrop', crop_size=crop_shape),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="Seg2T"),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='NoriDecode'),
    dict(type='Resize', scale=(-1, image_short)),
    dict(type='CenterCrop', crop_size=crop_shape),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="Seg2T"),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=32,
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
    type='SGD',
    lr=0.12,  # this lr is used for 1 gpus
    momentum=0.9,
    weight_decay=1e-6)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[14, 18])
total_epochs = 22
# work_dir = './work_dirs/tsm_r50_1x1x8_50e_sthv2_rgb_prototype/'
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', 
        log_dir="/data/mmaction2/tf_logs/ssl_test/sthv2/modist_coaug_samekn_posp_rm_mx_full1_lr2e-2_ssv2"),
    ])
