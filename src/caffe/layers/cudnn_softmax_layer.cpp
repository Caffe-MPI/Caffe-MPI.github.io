#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_softmax_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNSoftmaxLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  SoftmaxLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  // Initialize CUDNN.
  cudnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  cudnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_top_desc_);
  handles_setup_ = true;
}

template <typename Ftype, typename Btype>
void CuDNNSoftmaxLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  SoftmaxLayer<Ftype, Btype>::Reshape(bottom, top);
  int N = this->outer_num_;
  int K = bottom[0]->shape(this->softmax_axis_);
  int H = this->inner_num_;
  int W = 1;
  cudnn::setTensor4dDesc<Ftype>(&fwd_bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Ftype>(&fwd_top_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Btype>(&bwd_bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Btype>(&bwd_top_desc_, N, K, H, W);
}

template <typename Ftype, typename Btype>
CuDNNSoftmaxLayer<Ftype, Btype>::~CuDNNSoftmaxLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(fwd_bottom_desc_);
  cudnnDestroyTensorDescriptor(fwd_top_desc_);
  cudnnDestroyTensorDescriptor(bwd_bottom_desc_);
  cudnnDestroyTensorDescriptor(bwd_top_desc_);
}

INSTANTIATE_CLASS_FB(CuDNNSoftmaxLayer);

}  // namespace caffe
#endif
