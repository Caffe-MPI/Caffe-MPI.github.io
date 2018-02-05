#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/solvers/adam/solver_fp16.prototxt -gpu=all \
    2>&1 | tee examples/solvers/adam/adam_fp16.log
