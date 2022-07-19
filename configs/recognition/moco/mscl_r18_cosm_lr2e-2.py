_base_ = [
    '../../_base_/default_runtime.py'
]

# model settings
# s16_f8/rm_mx/cvqk/neg_without_upd/cosm
ft_dim = 128
image_shape = (112, 112)
num_frames = 8
stride = 8
crop_shape=128
total_epochs = 400
dataset_size=219136

rgb_recognizer = dict(
    type='MoCoV2',
    backbone=dict(
        type='torchvision.r3d_18',
        # pretrained='torchvision://resnet50', # set from work_dir
        ),
    neck = dict(
        type="TPNMoCo", in_channels=[128, 256, 512], out_channels=128,
        sepc_cfg=dict(in_channels=[128, 128, 128], out_channels=128, stride=(2, 2, 2), iBN=False, Pconv_num=2,),
    ),
    moco_head=dict(type="MoCoHead", 
                   basename='', loss_cls=dict(type="CrossEntropyLoss_torch", ignore_index=-1), ),
    im_key="imgs", dim_in=512, dim=ft_dim,
    # K=65536, m=0.999, T=0.07, mlp=True, aux_info=[], aug=dict(type="IdentityAug"), )
    K=65536, m_base=0.994, max_iters=dataset_size*total_epochs, T=0.07, mlp=True, aux_info=[], aug=dict(type="IdentityAug"), )
flow_recognizer = dict(
    type='MoCoV2',
    backbone=dict(
        type='resnet_flow.r2d_18',),
    neck = dict(
        type="BaseMoCo",
    ),
    moco_head=dict(type="MoCoHead", basename='flow', loss_cls=dict(type="CrossEntropyLoss_torch", ignore_index=-1)),
    im_key="imgs", dim_in=128, dim=ft_dim,
    # K=65536, m=0.999, T=0.07, mlp=True, aux_info=[], aug=dict(type="IdentityAug"), )
    K=65536, m_base=0.994, max_iters=dataset_size*total_epochs, T=0.07, mlp=True, aux_info=[], aug=dict(type="IdentityAug"), )
model = dict(
    type="MSCLWithAug",
    recognizer=rgb_recognizer, recognizer_flow=flow_recognizer, 
    moco_mx_head=dict(type="MSCLWithAugMxHead", basename='mx', loss_cls=dict(type="CrossEntropyLoss_torch", ignore_index=-1),
                      same_kn=True, T=0.07),   # div 2 for r2d_18
    sup_head=dict(type="MSCLWithAugPosHeadV2", basename='', loss_pos=dict(type="CrossEntropyLoss_torch", ignore_index=-1),
                  bkb_channels=(None, None), t=num_frames//2, T=0.07, 
                  aux_keys=dict(im_features=dict(q_mlvl='q_mlvl'), 
                                 base_flow_features=dict(q_mlvl='q_flow_mlvl'),
                                 aug_flow_features=dict(q_mlvl='q_aug_flow_mlvl'))),                    # neg without upd
    im_key='imgs', flow_key='flow_imgs', aux_info=[], update_aug_flow=False, weight_aug_flow=(1.0, 1.0),
    aug=dict(type="SyncMoCoAugmentV5", crop_size=image_shape[0], sync_level=('batch', 'batch'),
             t=(num_frames, num_frames), flow_suffix='flow_imgs', weak_aug=(False, False), visualize=True),
    same_kn=True,
)

# dataset settings
dataset_type = 'RedisRawframeDataset'
redis_url="redis://redis2.wuqian.ws2.hh-b.brainpp.cn:6379/1"
redis_master_url="redis://redis2.wuqian.ws2.hh-b.brainpp.cn:6379/1"
pkl_path = 's3://activity-public/kinetics400/flow_raft_ur/annos_v2/new/kinetics400_full_train_rm_mx_s4_merge.pkl'
pkl_path_val = 's3://activity-public/kinetics155/flow_raftur/annos/new/kinetics155_full_val_v3_rm_mx_s4.pkl'
extra_keys=["nids_flow", "chosen_idx"]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='MatchFlow', gap=2, adjacent=8, flow_key="nids_flow"),
    dict(type='TemporalShiftChosenSampleFrames', clip_len=num_frames, frame_interval=stride, num_clips=1, shift_range=1),
    dict(type='NoriDecode'),
    dict(type='NormFlowWithStidedAug', ratios=(0.2, 1.8), num_chunks=8, merge_aug=True),
    dict(type="MoCoRandomResizedCrop", area_range=(0.2, 1.0), flow_key="flow_imgs"),
    dict(type="MoCoResize", scale=image_shape, keep_ratio=False, flow_key="flow_imgs", suffix='_q'),
    dict(type="MoCoResize", scale=image_shape, keep_ratio=False, flow_key="flow_imgs", suffix='_k'),
    dict(type="MoCoNormalize", ori_flow=True),  # TODO: This should adjust to name for better project
    dict(type="Collect", keys=["imgs", "flow_imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "flow_imgs"], batched=True),
]
val_pipeline = [
    dict(type='MatchFlow', gap=2, adjacent=8, flow_key="nids_flow"),
    dict(type='ChosenSampleFrames', clip_len=num_frames, frame_interval=stride, num_clips=1),
    dict(type='NoriDecode'),
    dict(type='NormFlowWithStidedAug', ratios=(0.2, 1.8), num_chunks=8, merge_aug=True),
    dict(type="MoCoRandomResizedCrop", area_range=(0.2, 1.0), flow_key="flow_imgs"),
    dict(type="MoCoResize", scale=image_shape, keep_ratio=False, flow_key="flow_imgs", suffix='_q'),
    dict(type="MoCoResize", scale=image_shape, keep_ratio=False, flow_key="flow_imgs", suffix='_k'),
    dict(type="MoCoNormalize", ori_flow=True),  # TODO: This should adjust to name for better project
    dict(type="Collect", keys=["imgs", "flow_imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "flow_imgs"], batched=True),
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        redis_url=redis_url,
        redis_master_url=redis_master_url,
        pkl_path=pkl_path,
        pipeline=train_pipeline,
        extra_keys=extra_keys),
    val=dict(
        type=dataset_type,
        redis_url=redis_url,
        redis_master_url=redis_master_url,
        pkl_path=pkl_path_val,
        pipeline=val_pipeline,
        extra_keys=extra_keys),
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(drop_last=True),)

evaluation = dict(
    interval=5, simple=True)
# optimizer
# 128 -> 0.02, 256 -> 0.04
optimizer = dict(
    type='SGD',
    lr=0.02,  # this lr is used for 4 gpus, in large study, it is set to 0.4 for 512bs
    momentum=0.9,
    weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
# 5 will be set to warmup_epochs, then to self.warmup_iters = data_len*self.warmup_epochs
# in large study, lr is set to base*0.5*(cos(n/n_max*pi)+1)
lr_config = dict(policy='CosineAnnealing', min_lr=0, warmup_iters=5, warmup_by_epoch=True)
# runtime settings
# work_dir = None # This should be overlapped by args
work_dir = './work_dirs/ssl_train/modist_coaug_samekn_posp_rm_mx_full4_db_lr2e-2'
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir="/data/mmaction2/tf_logs/ssl_train/modist_coaug_samekn_posp_rm_mx_full4_db_lr2e-2"),
    ])
find_unused_parameters = True