#!/usr/bin/env bash

CFG=$1
CKPT=$2
OUT=$3      # output path for pkl, e.g. visualize/res_ssv2/moco_r18_lr3e-2.pkl

python tools/test_redis.py $CFG $CKPT --eval top_k_accuracy mean_class_accuracy --out $OUT