#!/usr/bin/env sh

./build/tools/caffe train --solver=models/resnet18/solver_fp16.prototxt -gpu=all \
    2>&1 | tee models/resnet18/logs/resnet18_fp16.log
