#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_googlenet/solver_fp16.prototxt -gpu=all \
    2>&1 | tee models/bvlc_googlenet/logs/googlenet_fp16.log
