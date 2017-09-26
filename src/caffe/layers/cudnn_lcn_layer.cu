#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lcn_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNLCNLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();

  temp1_.reserve(tempDataSize_);
  temp2_.reserve(tempDataSize_);

  CUDNN_CHECK(cudnnDivisiveNormalizationForward(
        Caffe::cudnn_handle(), norm_desc_, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
        cudnn::dataType<Ftype>::one,
        fwd_bottom_desc_, bottom_data,
        NULL,  // srcMeansData
        temp1_.data(), temp2_.data(),
        cudnn::dataType<Ftype>::zero,
        fwd_top_desc_, top_data) );
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));

  temp1_.release();
  temp2_.release();
}

template <typename Ftype, typename Btype>
void CuDNNLCNLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  const Btype* top_data = top[0]->gpu_data<Btype>();
  const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();

  temp1_.reserve(tempDataSize_);
  temp2_.reserve(tempDataSize_);

  CUDNN_CHECK(cudnnDivisiveNormalizationBackward(
        Caffe::cudnn_handle(), norm_desc_,
        CUDNN_DIVNORM_PRECOMPUTED_MEANS,
        cudnn::dataType<Btype>::one,
        bwd_bottom_desc_, bottom_data,
        NULL, top_diff,  // NULL - srcMeansData
        temp1_.data(), temp2_.data(),
        cudnn::dataType<Btype>::zero,
        bwd_bottom_desc_, bottom_diff,
        NULL) );
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));

  temp1_.release();
  temp2_.release();
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNLCNLayer);

}  // namespace caffe
#endif
