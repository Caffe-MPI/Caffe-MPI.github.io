#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_fp16.prototxt -gpu=all \
    2>&1 | tee models/bvlc_alexnet/logs/alexnet_fp16_2gpu.log
