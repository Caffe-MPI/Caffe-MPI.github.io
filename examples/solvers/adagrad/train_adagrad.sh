#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/solvers/adagrad/solver.prototxt -gpu=all \
    2>&1 | tee examples/solvers/adagrad/adagrad.log
