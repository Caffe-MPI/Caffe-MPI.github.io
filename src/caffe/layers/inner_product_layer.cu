#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void InnerProductLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const Ftype* weight = this->blobs_[0]->template gpu_data<Ftype>();
  // Y = X * W
  if (M_ == 1) {
    caffe_gpu_gemv<Ftype>(CblasNoTrans, N_, K_, (Ftype)1., weight, bottom_data,
        (Ftype)0., top_data);
  } else {
    caffe_gpu_gemm<Ftype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans, M_, N_, K_,
        (Ftype)1., bottom_data, weight, (Ftype)0., top_data);
  }
  // Y = Y + Bias(c)
  if (bias_term_) {
    const Ftype* bias = this->blobs_[1]->template gpu_data<Ftype>();
    if (M_ == 1) {
      caffe_gpu_axpy<Ftype>(N_, bias_multiplier_->template cpu_data<Ftype>()[0], bias, top_data);
    } else {
      caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Ftype)1.,
          bias_multiplier_->template gpu_data<Ftype>(), bias, (Ftype)1., top_data);
    }
  }
}

template <typename Ftype, typename Btype>
void InnerProductLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  // dE/dW and dE/dB
  // dE/dW: Gradient with respect to weight
  if (this->param_propagate_down_[0]) {
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    Btype* weight_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
    // dW = dY * X
    const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
    if (transpose_) {
      caffe_gpu_gemm<Btype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Btype)1., bottom_data, top_diff, (Btype)1., weight_diff);
    } else {
      caffe_gpu_gemm<Btype>(CblasTrans, CblasNoTrans, N_, K_, M_,
          (Btype)1., top_diff, bottom_data, (Btype)1., weight_diff);
    }
  }
  // dB: Gradient with respect to bias
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    Btype* bias_diff = this->blobs_[1]->template mutable_gpu_diff<Btype>();
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    // dB (c) = sum_N(dY(n, c))
    caffe_gpu_gemv<Btype>(CblasTrans, M_, N_, (Btype)1., top_diff,
        bias_multiplier_->template gpu_data<Btype>(), (Btype)1., bias_diff);
  }
  // Backward propagate dE/dX= dE/dY * W
  if (propagate_down[0]) {
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    const Btype* weight = this->blobs_[0]->template gpu_data<Btype>();
    // regular backward
    if (transpose_) {
      caffe_gpu_gemm<Btype>(CblasNoTrans, CblasTrans, M_, K_, N_,
         (Btype)1., top_diff, weight,
         (Btype)0., bottom[0]->mutable_gpu_diff<Btype>());
    } else {
      caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, M_, K_, N_,
         (Btype)1., top_diff, weight,
         (Btype)0., bottom[0]->mutable_gpu_diff<Btype>());
    }
  }  // end of regular backward
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(InnerProductLayer);

}  // namespace caffe
