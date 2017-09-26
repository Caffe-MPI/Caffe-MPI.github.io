#!/usr/bin/env sh

./build/tools/caffe train --solver=models/inception_v3/solver_fp16.prototxt -gpu=all \
    2>&1 | tee models/inception_v3/logs/inception_v3_fp16.log
