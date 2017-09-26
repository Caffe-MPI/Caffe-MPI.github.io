#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_tanh_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNTanHLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  TanHLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  // initialize cuDNN
  cudnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  cudnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_top_desc_);
  cudnnCreateActivationDescriptor(&activ_desc_);
  cudnnSetActivationDescriptor(activ_desc_, CUDNN_ACTIVATION_TANH,
    CUDNN_NOT_PROPAGATE_NAN, 0.0);
  handles_setup_ = true;
}

template <typename Ftype, typename Btype>
void CuDNNTanHLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  TanHLayer<Ftype, Btype>::Reshape(bottom, top);
  const int N = bottom[0]->num();
  const int K = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  cudnn::setTensor4dDesc<Ftype>(&fwd_bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Ftype>(&fwd_top_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Btype>(&bwd_bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Btype>(&bwd_top_desc_, N, K, H, W);
}

template <typename Ftype, typename Btype>
CuDNNTanHLayer<Ftype, Btype>::~CuDNNTanHLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyActivationDescriptor(this->activ_desc_);
  cudnnDestroyTensorDescriptor(fwd_bottom_desc_);
  cudnnDestroyTensorDescriptor(fwd_top_desc_);
  cudnnDestroyTensorDescriptor(bwd_bottom_desc_);
  cudnnDestroyTensorDescriptor(bwd_top_desc_);
}

INSTANTIATE_CLASS_FB(CuDNNTanHLayer);

}  // namespace caffe
#endif
