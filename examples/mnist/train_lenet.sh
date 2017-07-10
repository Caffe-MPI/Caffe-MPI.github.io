#!/usr/bin/env sh

#./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt --gpu=0 2>&1 |tee log
cuda-gdb ./build/tools/caffe
