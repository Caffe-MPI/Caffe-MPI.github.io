#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/solvers/adadelta/solver_fp16.prototxt -gpu=all \
    2>&1 | tee examples/solvers/adadelta/adadelta_fp16.log
