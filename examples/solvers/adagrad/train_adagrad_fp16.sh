#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/solvers/adagrad/solver_fp16.prototxt -gpu=all \
    2>&1 | tee examples/solvers/adagrad/adagrad_fp16.log
