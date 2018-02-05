#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lrn_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNLRNLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  LRNLayer<Ftype, Btype>::LayerSetUp(bottom, top);

  CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc_));
  cudnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  cudnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_top_desc_);

  // create a LRN handle
  handles_setup_ = true;

  this->size_ = this->layer_param().lrn_param().local_size();
  this->alpha_ = this->layer_param().lrn_param().alpha();
  this->beta_ = this->layer_param().lrn_param().beta();
  this->k_ = this->layer_param().lrn_param().k();
}

template <typename Ftype, typename Btype>
void CuDNNLRNLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  LRNLayer<Ftype, Btype>::Reshape(bottom, top);
  cudnn::setTensor4dDesc<Ftype>(&fwd_bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc<Ftype>(&fwd_top_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc<Btype>(&bwd_bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc<Btype>(&bwd_top_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc_,
      this->size_, this->alpha_, this->beta_, this->k_));
}

template <typename Ftype, typename Btype>
CuDNNLRNLayer<Ftype, Btype>::~CuDNNLRNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(fwd_bottom_desc_);
  cudnnDestroyTensorDescriptor(fwd_top_desc_);
  cudnnDestroyTensorDescriptor(bwd_bottom_desc_);
  cudnnDestroyTensorDescriptor(bwd_top_desc_);

  // destroy LRN handle
  CUDNN_CHECK(cudnnDestroyLRNDescriptor(norm_desc_));
}

INSTANTIATE_CLASS_FB(CuDNNLRNLayer);

}   // namespace caffe

#endif
