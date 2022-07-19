#!/usr/bin/env bash

CONFIG=$1
CKPT=$2
python tools/test_retrival.py $CONFIG $CKPT --ssl