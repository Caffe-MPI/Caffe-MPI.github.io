#!/usr/bin/env sh

./build/tools/caffe train --solver=models/inception_v2/solver.prototxt -gpu=all \
    2>&1 | tee models/inception_v2/logs/inception_v2.log
