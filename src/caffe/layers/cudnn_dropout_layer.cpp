#ifdef USE_CUDNN

#include "caffe/layers/cudnn_dropout_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNDropoutLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  DropoutLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  // initialize cuDNN
  cudnn::createTensor4dDesc<Ftype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Ftype>(&top_desc_);

  // initialize dropout state
  CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
  CUDNN_CHECK(cudnnDropoutGetStatesSize(Caffe::cudnn_handle(), &state_size_));
  states_.reserve(state_size_);

  // setup dropout descriptor
  CUDNN_CHECK(cudnnSetDropoutDescriptor(dropout_desc_,
                                        Caffe::cudnn_handle(),
                                        this->threshold_,
                                        states_.data(),
                                        state_size_,
                                        seed_));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));

  handles_setup_ = true;
}

template <typename Ftype, typename Btype>
void CuDNNDropoutLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  DropoutLayer<Ftype, Btype>::Reshape(bottom, top);
  const int N = bottom[0]->num();
  const int K = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  cudnn::setTensor4dDesc<Ftype>(&bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Ftype>(&top_desc_, N, K, H, W);

  CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(bottom_desc_,
                                              &reserve_space_size_));
  reserve_space_.reserve(reserve_space_size_);
}

template <typename Ftype, typename Btype>
CuDNNDropoutLayer<Ftype, Btype>::~CuDNNDropoutLayer() {
  states_.release();
  reserve_space_.release();
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }
  cudnnDestroyTensorDescriptor(this->bottom_desc_);
  cudnnDestroyTensorDescriptor(this->top_desc_);

  CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
}

INSTANTIATE_CLASS_FB(CuDNNDropoutLayer);

}  // namespace caffe
#endif
