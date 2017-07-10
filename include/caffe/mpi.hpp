#ifndef CAFFE_UTIL_MPI_H_
#define CAFFE_UTIL_MPI_H_

#include <mpi.h>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

template <typename Dtype>
int caffe_mpi_allreduce(void *buf1,void *buf2, int count, 
                    MPI_Comm comm);

int caffe_mpi_allreduce(void *buf1,void *buf2, int count, MPI_Datatype datatype, MPI_Op ops,
                    MPI_Comm comm);

}  // namespace caffe

#endif  // CAFFE_UTIL_MPI_H_
