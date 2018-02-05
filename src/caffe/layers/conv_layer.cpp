#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ConvolutionLayer<Ftype, Btype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Ftype, typename Btype>
void ConvolutionLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const Ftype* weight = this->blobs_[0]->template cpu_data<Ftype>();
  for (int i = 0; i < bottom.size(); ++i) {
    const Ftype* bottom_data = bottom[i]->cpu_data<Ftype>();
    Ftype* top_data = top[i]->mutable_cpu_data<Ftype>();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Ftype* bias = this->blobs_[1]->template cpu_data<Ftype>();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Ftype, typename Btype>
void ConvolutionLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const Btype* weight = this->blobs_[0]->template cpu_data<Btype>();
  Btype* weight_diff = this->blobs_[0]->template mutable_cpu_diff<Btype>();
  for (int i = 0; i < top.size(); ++i) {
    const Btype* top_diff = top[i]->cpu_diff<Btype>();
    const Btype* bottom_data = bottom[i]->cpu_data<Btype>();
    Btype* bottom_diff = bottom[i]->mutable_cpu_diff<Btype>();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Btype* bias_diff = this->blobs_[1]->template mutable_cpu_diff<Btype>();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS_FB(ConvolutionLayer);

}  // namespace caffe
