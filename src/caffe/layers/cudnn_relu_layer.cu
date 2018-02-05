#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_relu_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNReLULayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
     const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  // Fallback to standard Caffe for leaky ReLU.
  if (ReLULayer<Ftype, Btype>::layer_param_.relu_param().negative_slope() != 0) {
    return ReLULayer<Ftype, Btype>::Forward_gpu(bottom, top);
  }
  CUDNN_CHECK(cudnnActivationForward(Caffe::cudnn_handle(),
        activ_desc_,
        cudnn::dataType<Ftype>::one,
        fwd_bottom_desc_, bottom_data,
        cudnn::dataType<Ftype>::zero,
        fwd_top_desc_, top[0]->mutable_gpu_data<Ftype>()));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

template <typename Ftype, typename Btype>
void CuDNNReLULayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Btype* top_data = top[0]->gpu_data<Btype>();
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
  // Fallback to standard Caffe for leaky ReLU.
  if (ReLULayer<Ftype, Btype>::layer_param_.relu_param().negative_slope() != 0) {
    return ReLULayer<Ftype, Btype>::Backward_gpu(top, propagate_down, bottom);
  }
  CUDNN_CHECK(cudnnActivationBackward(Caffe::cudnn_handle(),
        activ_desc_,
        cudnn::dataType<Btype>::one,
        bwd_top_desc_, top_data, bwd_top_desc_, top_diff,
        bwd_bottom_desc_, bottom_data,
        cudnn::dataType<Btype>::zero,
        bwd_bottom_desc_, bottom_diff));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNReLULayer);

}  // namespace caffe
#endif
