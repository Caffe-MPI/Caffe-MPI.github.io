#!/bin/sh
#mpiexec -env I_MPI_FABRICS=shm:ofa -prepend-rank -machinefile hostsib -n 3 ./build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt 2>&1 | tee test.log
mpiexec -env I_MPI_FABRICS=shm:ofa -machinefile hostsib -env MV2_ENABLE_AFFINITY 0 -env MV2_USE_CUDA 1 -env MV2_USE_GPUDIRECT 1 \
 -prepend-rank  -n 20 ./build/tools/caffe train \
 --solver=examples/cifar10/cifar10_quick_solver.prototxt 2>&1 | tee test.log
#mpirun --allow-run-as-root -report-pid - -machinefile hostsib1 -n 3 ./build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt 2>&1 | tee test.log
