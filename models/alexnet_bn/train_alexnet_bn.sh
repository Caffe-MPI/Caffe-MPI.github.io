#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/alexnet_bn/solver.prototxt -gpu=all \
    2>&1 | tee models/alexnet_bn/logs/alexnet_bn_globalvar.log
