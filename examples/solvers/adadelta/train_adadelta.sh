#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/solvers/adadelta/solver.prototxt -gpu=all \
    2>&1 | tee examples/solvers/adadelta/adadelta.log
