#include <vector>

#include "caffe/layers/deconv_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void DeconvolutionLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const Ftype* weight = this->blobs_[0]->template gpu_data<Ftype>();
  for (int i = 0; i < bottom.size(); ++i) {
    const Ftype* bottom_data = bottom[i]->gpu_data<Ftype>();
    Ftype* top_data = top[i]->mutable_gpu_data<Ftype>();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Ftype* bias = this->blobs_[1]->template gpu_data<Ftype>();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Ftype, typename Btype>
void DeconvolutionLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const Btype* weight = this->blobs_[0]->template gpu_data<Btype>();
  Btype* weight_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
  for (int i = 0; i < top.size(); ++i) {
    const Btype* top_diff = top[i]->gpu_diff<Btype>();
    const Btype* bottom_data = bottom[i]->gpu_data<Btype>();
    Btype* bottom_diff = bottom[i]->mutable_gpu_diff<Btype>();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Btype* bias_diff = this->blobs_[1]->template mutable_gpu_diff<Btype>();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(top_diff + n * this->top_dim_,
              bottom_data + n * this->bottom_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->forward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_,
              this->param_propagate_down_[0]);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(DeconvolutionLayer);

}  // namespace caffe
