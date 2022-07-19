_base_ = [
    '../../_base_/default_runtime.py'
]

# model settings
ft_dim = 128
image_shape = (112, 112)
image_short = 256
num_frames = 8
stride = 8
model = dict(
    type='MoCo',
    backbone=dict(
        type='torchvision.r3d_18',
        # pretrained='torchvision://resnet50', # set from work_dir
        ),
    neck = dict(
        type="BaseMoCo",
    ),
    moco_head=dict(type="MoCoHead", loss_cls=dict(type="CrossEntropyLoss_torch", ignore_index=-1)),
    im_key="imgs", dim_in=512, dim=ft_dim,
    K=65536,
    m=0.999,
    T=0.07,
    mlp=True,
    aux_info=[],
    aug=dict(type="MoCoAugmentV2", crop_size=image_shape[0]), )
# changeable parameters
image_shape=(112, 112)
crop_shape=128
# dataset settings
dataset_type = 'RedisRawframeDataset'
redis_url="redis://redis2.wuqian.ws2.hh-b.brainpp.cn:6379/1"
redis_master_url="redis://redis2.wuqian.ws2.hh-b.brainpp.cn:6379/1"
pkl_path = 's3://activity-public/kinetics600/annos/kinetics155_temporal_train.pkl'
pkl_path_val = 's3://activity-public/kinetics600/annos/kinetics155_temporal_val.pkl'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=num_frames, frame_interval=stride, num_clips=1),
    dict(type='NoriDecode'),
    dict(
        type="MoCoTransform",
        transform=[dict(type='ToTensorVideo'),],  # n,c,h,w -> c,n,h,w + norm(div255)
        crop_transform=dict(size=image_shape, scale=(0.2, 1)),
        ending_transform=[],
    ),
    dict(type="Collect", keys=["imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"], batched=True),
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=num_frames, frame_interval=stride, num_clips=1),
    dict(type='NoriDecode'),
    dict(
        type="MoCoTransform",
        transform=[dict(type='ToTensorVideo'),],  # n,c,h,w -> c,n,h,w + norm(div255)
        crop_transform=dict(size=image_shape, scale=(0.2, 1)),
        ending_transform=[],
    ),
    dict(type="Collect", keys=["imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"], batched=True),
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        redis_url=redis_url,
        redis_master_url=redis_master_url,
        pkl_path=pkl_path,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        redis_url=redis_url,
        redis_master_url=redis_master_url,
        pkl_path=pkl_path_val,
        pipeline=val_pipeline),
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(drop_last=True),)

evaluation = dict(
    interval=5, simple=True)
# optimizer
# 128 -> 0.02, 256 -> 0.04
optimizer = dict(
    type='SGD',
    lr=0.015,  # this lr is used for 4 gpus, in large study, it is set to 0.4 for 512bs
    momentum=0.9,
    weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
# 5 will be set to warmup_epochs, then to self.warmup_iters = data_len*self.warmup_epochs
# in large study, lr is set to base*0.5*(cos(n/n_max*pi)+1)
lr_config = dict(policy='CosineAnnealing', min_lr=0, warmup_iters=5, warmup_by_epoch=True)
total_epochs = 120
# runtime settings
# work_dir = None # This should be overlapped by args
work_dir = './work_dirs/ssl_train/moco_base_lr3e-2'
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir="/data/mmaction2/tf_logs/ssl_train/moco_base"),
    ])
# find_unused_parameters = True