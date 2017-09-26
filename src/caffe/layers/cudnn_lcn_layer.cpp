#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lcn_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNLCNLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  LRNLayer<Ftype, Btype>::LayerSetUp(bottom, top);

  CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc_));
  cudnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  cudnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_top_desc_);

  // create a LRN handle
  handles_setup_ = true;

  size_ = this->layer_param().lrn_param().local_size();
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param().lrn_param().alpha();
  beta_ = this->layer_param().lrn_param().beta();
  k_ = this->layer_param().lrn_param().k();
}

template <typename Ftype, typename Btype>
void CuDNNLCNLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
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
  CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, k_));

  // size for temp buffers
  tempDataSize_ = sizeof(Ftype) * bottom[0]->num() * this->channels_ *
      this->height_ * this->width_;
}

template <typename Ftype, typename Btype>
CuDNNLCNLayer<Ftype, Btype>::~CuDNNLCNLayer() {
  temp1_.release();
  temp2_.release();

  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  CUDNN_CHECK(cudnnDestroyTensorDescriptor(fwd_bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(fwd_top_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bwd_bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bwd_top_desc_));

  // destroy LRN handle
  CUDNN_CHECK(cudnnDestroyLRNDescriptor(norm_desc_));
}

INSTANTIATE_CLASS_FB(CuDNNLCNLayer);

}   // namespace caffe
#endif
