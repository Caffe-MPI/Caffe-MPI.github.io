#include "caffe/common.hpp"
#include "caffe/util/mpi.hpp"

#include <execinfo.h>
namespace caffe {

template<>
int caffe_mpi_send<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm) {
/*
int j, nptrs;
 void *buffer[100];
   char **strings;
nptrs = backtrace(buffer, 3);
strings = backtrace_symbols(buffer, nptrs);
for (j = 0; j < nptrs; j++)

       printf("%s\n", strings[j]);

   free(strings);
*/
//	LOG(INFO)<<"MPI_SEND "<<buf<<" "<<count<<" "<<dest<<" "<<tag<<" ";
//	int size=1024*1024*1024*1;
//	char * bbuf= new char[size];
//        MPI_Buffer_attach((void*)bbuf,size);
	int ret=MPI_Send(buf, count, MPI_FLOAT, dest, tag,
                    comm);
//	MPI_Buffer_detach((void*)bbuf,&size);
	return ret;
}

template<>
int caffe_mpi_send<double>(void *buf, int count,  int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Send(buf, count, MPI_DOUBLE, dest, tag,
                    comm);
}

int caffe_mpi_send(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Send(buf, count, datatype, dest, tag,
                    comm);
}
template<>
int caffe_mpi_recv<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
	//LOG(INFO)<<"MPI_RECV "<<buf<<" "<<count<<" "<<dest<<" "<<tag<<" ";
	int ret=MPI_Recv(buf, count, MPI_FLOAT, dest, tag,
                    comm, status);
	return ret;
}

template<>
int caffe_mpi_recv<double>(void *buf, int count,  int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
	return MPI_Recv(buf, count, MPI_DOUBLE, dest, tag,
                    comm, status);
}

int caffe_mpi_recv(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
	return MPI_Recv(buf, count, datatype, dest, tag,
                    comm, status);
}

template <>
int caffe_mpi_isend<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
	return MPI_Isend(buf, count, MPI_FLOAT, dest, tag,comm, req);
}

template <>
int caffe_mpi_isend<double>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
	return MPI_Isend(buf, count, MPI_DOUBLE, dest, tag,comm, req);
}

int caffe_mpi_isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
	return MPI_Isend(buf, count, datatype, dest, tag,comm, req);
}
template <>
int caffe_mpi_ssend<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Ssend(buf, count, MPI_FLOAT, dest, tag,comm);
}

template <>
int caffe_mpi_ssend<double>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Ssend(buf, count, MPI_DOUBLE, dest, tag,comm);
}

int caffe_mpi_ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Ssend(buf, count, datatype, dest, tag,comm);
}

}  // namespace caffe
