#!/usr/bin/env sh

./build/tools/caffe train --solver=models/vgg16/solver.prototxt -gpu=all \
    2>&1 | tee models/vgg16/logs/vgg16.log
