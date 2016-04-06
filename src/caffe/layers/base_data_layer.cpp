#include <boost/thread.hpp>
#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  //data_transformer_->InitRand();
  // LOG(INFO)<<"TAG_DATA_recv"<<rank<<" "<<top[0]->gpu_data()[0];
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  CHECK_GT(datum_channels_, 0);
  CHECK_GT(datum_height_, 0);
  CHECK_GT(datum_width_, 0);
  if (transform_param_.crop_size() > 0) {
    CHECK_GE(datum_height_, transform_param_.crop_size());
    CHECK_GE(datum_width_, transform_param_.crop_size());
  }
  if (transform_param_.has_mean_file()) {
    const string& mean_file = transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_GE(data_mean_.num(), 1);
    CHECK_GE(data_mean_.channels(), datum_channels_);
    CHECK_GE(data_mean_.height(), datum_height_);
    CHECK_GE(data_mean_.width(), datum_width_);
  } else {
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  mean_ = data_mean_.cpu_data();
  data_transformer_->InitRand();


}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_() {
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
 // if(rank ==0 ){
   for (int i = 0; i < PREFETCH_COUNT; ++i) {
     prefetch_free_.push(&prefetch_[i]);
   }
 // }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
 /*  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
if(Caffe::phase()==Caffe::TEST){//TODO
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  switch (this->layer_param_.data_param().backend()){
          case DataParameter_DB_LEVELDB:
                  {
                          if(rank==0){
                                  this->CreatePrefetchThread();
                                  DLOG(INFO) << "Prefetch initialized.";
                          }
                  }
                  break;
          case DataParameter_DB_LMDB:
                  {
                          if(rank==0){
                                  this->CreatePrefetchThread();
                                  DLOG(INFO) << "Prefetch initialized.";
                          }
break;
          default:
                  LOG(FATAL) << "Unknown database backend";
  }
}
}
   */
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.

  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
  }


#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
    }
  }
#endif

  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        //LOG(INFO)<<"pushing to gpu";
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,const  vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  /*
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }
  */
 #ifndef ASYNCTRAN
        MPI_Status status;
        status.MPI_ERROR=0;
      //  LOG(INFO)<<"before recv"<<top[0]->mutable_cpu_data()[0];
        caffe_mpi_recv<Dtype>((top)[0]->mutable_cpu_data(),prefetch_data_.count(),
                        0,TAG_DATA_OUT,MPI_COMM_WORLD,&status);
        DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
	//LOG(INFO)<<"after recv"<<top[0]->mutable_cpu_data()[0];
        if (this->output_labels_) {
                caffe_mpi_recv<Dtype>((top)[1]->mutable_cpu_data(),prefetch_label_.count(),
                                0,TAG_DATA_OUT_IF,MPI_COMM_WORLD,&status);
                DLOG(INFO)<<"Recv Dataout if status "<<status.MPI_ERROR;
        }
#endif
  prefetch_free_.push(batch);
}

//added by zhuhui 20151207
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu_test(
    const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
  JoinPrefetchThread();
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             (top)[0]->mutable_cpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               (top)[1]->mutable_cpu_data());
  }
  CreatePrefetchThread();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ =static_cast<caffe::Phase>( Caffe::phase());
  //this->data_transformer_.InitRand();
  //CHECK(StartInternalThread()) << "Thread execution failed";
   StartInternalThread();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
template <typename Dtype> \
void BasePrefetchingDataLayer<Dtype>::Forward_gpu_test(const vector<Blob<Dtype>*>& bottom, \
const    vector<Blob<Dtype>*>& top) { NO_GPU; }
#endif








INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
