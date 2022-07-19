# TSM
# bash ./tools/dist_train.sh configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb_triplet.py 8 \
#     # --resume-from work_dirs/tsm_r50_1x1x8_50e_sthv2_rgb_triplet/epoch_16.pth 
#     --validate --seed 0 --deterministic

# SSL
bash ./tools/dist_train.sh configs/recognition/resim_dir/base_lr2e-2_no_rev.py 4 \
    --validate --seed 0 --deterministic
    # --resume-from work_dirs/tsm_r50_1x1x8_50e_sthv2_rgb_triplet/epoch_16.pth 