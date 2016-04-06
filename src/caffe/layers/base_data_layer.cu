#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/mpi.hpp"
namespace caffe {
template <typename Dtype>

void BasePrefetchingDataLayer<Dtype>::Forward_gpu_test(
    const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
  for(int i=0;i<PREFETCH_COUNT;++i){
  JoinPrefetchThread();
  caffe_copy(prefetch_[i].data_.count(), prefetch_[i].data_.cpu_data(),
      (top)[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_[i].label_.count(), prefetch_[i].label_.cpu_data(),
        (top)[1]->mutable_gpu_data());
  }
  CreatePrefetchThread();
 }
}


template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  top[0]->ReshapeLike(batch->data_);
#ifndef ASYNCTRAN
DBGPRT(LOG(INFO)<<"RECV DATA");
        MPI_Status status;
        status.MPI_ERROR=0;
	int tid33;

if(rank<10)
{
	if(rank <5) 
		tid33 =0;
	else
		tid33 = 5;
}
else 
{
	if(rank<15)
		tid33 =10;
	else
		tid33 = 15;
}
#ifdef DIRECTGPU
        caffe_mpi_recv<Dtype>((top)[0]->mutable_gpu_data(),(top)[0]->count(),
                        tid33,TAG_DATA_OUT,MPI_COMM_WORLD,&status);
        DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
        if (this->output_labels_) {
                caffe_mpi_recv<Dtype>((top)[1]->mutable_gpu_data(),(top)[1]->count(),
                                tid33,TAG_DATA_OUT_IF,MPI_COMM_WORLD,&status);
                DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
        }

#else
       caffe_mpi_recv<Dtype>((top)[0]->mutable_cpu_data(),(top)[0]->count(),
                        0,TAG_DATA_OUT,MPI_COMM_WORLD,&status);
        DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
        if (this->output_labels_) {
                caffe_mpi_recv<Dtype>((top)[1]->mutable_cpu_data(),(top)[1]->count(),
                                0,TAG_DATA_OUT_IF,MPI_COMM_WORLD,&status);
                DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
        }
#endif

DBGPRT(LOG(INFO)<<"RECV DATA FIN");
#endif
 
 // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
  template void BasePrefetchingDataLayer<float>::Forward_gpu_test( \
      const std::vector<Blob<float>*>& bottom, \
      const std::vector<Blob<float>*>& top); \
  template void BasePrefetchingDataLayer<double>::Forward_gpu_test( \
      const std::vector<Blob<double>*>& bottom, \
      const std::vector<Blob<double>*>& top);


}  // namespace caffe
