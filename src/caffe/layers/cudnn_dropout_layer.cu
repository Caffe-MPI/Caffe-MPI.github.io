#ifdef USE_CUDNN

#include <vector>

#include "caffe/layers/cudnn_dropout_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNDropoutLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  if (this->phase_ == TRAIN) {
    CUDNN_CHECK(cudnnDropoutForward(Caffe::cudnn_handle(),
        dropout_desc_,
        this->bottom_desc_, bottom_data,
        this->top_desc_, top_data,
        reserve_space_.data(), reserve_space_size_));
    CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
  } else {
    caffe_copy<Ftype>(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Ftype, typename Btype>
void CuDNNDropoutLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
  if (propagate_down[0]) {
    if (this->phase_ == TRAIN) {
      CUDNN_CHECK(cudnnDropoutBackward(Caffe::cudnn_handle(),
          dropout_desc_,
          this->top_desc_, top_diff,
          this->bottom_desc_, bottom_diff,
          reserve_space_.data(), reserve_space_size_));
      CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNDropoutLayer);

}  // namespace caffe
#endif
