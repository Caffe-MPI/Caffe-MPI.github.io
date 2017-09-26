#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_pooling_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNPoolingLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();

  CUDNN_CHECK(cudnnPoolingForward(Caffe::cudnn_handle(), pooling_desc_,
      cudnn::dataType<Ftype>::one, fwd_bottom_desc_, bottom_data,
      cudnn::dataType<Ftype>::zero, fwd_top_desc_, top_data));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));

  if (this->is_max_pooling_) {
    for (int i = 0; i < top.size(); ++i) {
      private_top_[i]->CopyDataFrom(*top[i], true);
    }
  }
}

template <typename Ftype, typename Btype>
void CuDNNPoolingLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();

  Blob* top_blob = this->is_max_pooling_ ? private_top_[0].get() : top[0];
  const Btype* top_data = top_blob->gpu_data<Btype>();

  CUDNN_CHECK(cudnnPoolingBackward(Caffe::cudnn_handle(),  pooling_desc_,
      cudnn::dataType<Btype>::one, bwd_top_desc_, top_data, bwd_top_desc_, top_diff,
      bwd_bottom_desc_, bottom_data, cudnn::dataType<Btype>::zero, bwd_bottom_desc_, bottom_diff));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNPoolingLayer);

}  // namespace caffe
#endif
