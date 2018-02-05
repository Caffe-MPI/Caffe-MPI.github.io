#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lrn_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNLRNLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  CUDNN_CHECK(cudnnLRNCrossChannelForward(Caffe::cudnn_handle(),
      norm_desc_,
      CUDNN_LRN_CROSS_CHANNEL_DIM1,
      cudnn::dataType<Ftype>::one,
      fwd_bottom_desc_, bottom[0]->gpu_data<Ftype>(),
      cudnn::dataType<Ftype>::zero,
      fwd_top_desc_, top[0]->mutable_gpu_data<Ftype>()));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

template <typename Ftype, typename Btype>
void CuDNNLRNLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  CUDNN_CHECK(cudnnLRNCrossChannelBackward(Caffe::cudnn_handle(),
      norm_desc_,
      CUDNN_LRN_CROSS_CHANNEL_DIM1,
      cudnn::dataType<Btype>::one,
      bwd_top_desc_, top[0]->gpu_data<Btype>(),
      bwd_top_desc_, top[0]->gpu_diff<Btype>(),
      bwd_bottom_desc_, bottom[0]->gpu_data<Btype>(),
      cudnn::dataType<Btype>::zero,
      bwd_bottom_desc_, bottom[0]->mutable_gpu_diff<Btype>()));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNLRNLayer);

}  // namespace caffe
#endif
