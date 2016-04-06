#ifndef CAFFE_UTIL_MPI_H_
#define CAFFE_UTIL_MPI_H_

#include <mpi.h>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

template <typename Dtype>
int caffe_mpi_send(void *buf, int count, int dest, int tag,
                    MPI_Comm comm);

int caffe_mpi_send(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm);

template <typename Dtype>
int caffe_mpi_recv(void *buf, int count,  int source, int tag,
                    MPI_Comm comm, MPI_Status *status);

int caffe_mpi_recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                    MPI_Comm comm, MPI_Status *status);

template <typename Dtype>
int caffe_mpi_isend(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req);

int caffe_mpi_isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req);

template <typename Dtype>
int caffe_mpi_ssend(void *buf, int count, int dest, int tag,
                    MPI_Comm comm);

int caffe_mpi_ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm);
}  // namespace caffe

#endif  // CAFFE_UTIL_MPI_H_
