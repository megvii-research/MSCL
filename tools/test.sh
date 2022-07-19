CFG=configs/recognition/ssl_test/templete_r3d50_fmoco_k400_modist_nodropout.py
CKPT=work_dirs/ssl_test/k400/modist_coaug_samekn_posp_rm_mx_full5_r50_lr2e-2_nodropout/epoch_50.pth
python tools/test.py $CFG $CKPT --eval top_k_accuracy mean_class_accuracy \
    --out visualize/k400/modist_coaug_samekn_posp_rm_mx_full5_r50_lr2e-2_nodropout.pkl