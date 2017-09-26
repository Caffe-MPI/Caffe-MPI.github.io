#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/cifar10_nv/cifar10_nv_solver.prototxt  -gpu=all \
    2>&1 | tee models/cifar10_nv/logs/cifar10_nv.log


