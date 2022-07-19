CFG=configs/recognition/ssl_test/templete_r3d50_fmoco_sthv2_modist.py
# CKPT=work_dirs/ssl_test/sthv2/moco_consistent_aug_lr3e-2_k155/epoch_20.pth
CKPT=work_dirs/ssl_test/sthv2/modist_coaug_samekn_posp_rm_mx_full5_r50_lr2e-2_ssv2/epoch_20.pth
MPTH=visualize/tgt_inds/posp_modistk155_relonly.pkl
python demo/demo_gradcam_mscl.py $CFG $CKPT --use-frames --out-filename /data/visualize/mscl/split/test/frame%04d.jpeg
#--metapath $MPTH