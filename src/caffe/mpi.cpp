#include "caffe/common.hpp"
#include "caffe/mpi.hpp"

#include <execinfo.h>
namespace caffe {

template<>
int caffe_mpi_allreduce<float>(void *buf1,void *buf2, int count, 
                    MPI_Comm comm) {

	int ret=MPI_Allreduce(buf1, buf2,count, MPI_FLOAT, MPI_SUM,
                    comm);

	return ret;
}

template<>
int caffe_mpi_allreduce<double>(void *buf1,void *buf2, int count,
                    MPI_Comm comm) {
	return MPI_Allreduce(buf1, buf2, count, MPI_DOUBLE,  MPI_SUM,
                    comm);
}

int caffe_mpi_allreduce(void *buf1,void *buf2, int count, MPI_Datatype datatype, MPI_Op ops,
                    MPI_Comm comm) {
	return MPI_Allreduce(buf1, buf2, count, datatype,ops,
                    comm);
}

}  // namespace caffe
