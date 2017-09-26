#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/alexnet_owt/solver_fp16.prototxt -gpu=all \
    2>&1 | tee models/alexnet_owt/logs/alexnet_owt_fp16.log
