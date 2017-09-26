#!/usr/bin/env sh

./build/tools/caffe train --solver=models/vgg16/solver_fp16.prototxt -gpu=all \
    2>&1 | tee models/vgg16/logs/vgg16_fp16.log
