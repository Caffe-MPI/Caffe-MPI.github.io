#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BiasForward(const int n, const Dtype* in,
    const Dtype* bias, const int bias_dim, const int inner_dim,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int bias_index = (index / inner_dim) % bias_dim;
    out[index] = in[index] + bias[bias_index];
  }
}

template <typename Ftype, typename Btype>
void BiasLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const int count = top[0]->count();
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  const Ftype* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->template gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  BiasForward  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
      count, bottom_data, bias_data, bias_dim_, inner_dim_, top_data);
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

template <typename Ftype, typename Btype>
void BiasLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) {
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    Btype* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->template mutable_gpu_diff<Btype>();
    bool accum = bias_param;
    for (int n = 0; n < outer_dim_; ++n) {
      caffe_gpu_gemv(CblasNoTrans, bias_dim_, inner_dim_, Btype(1),
          top_diff, bias_multiplier_.template gpu_data<Btype>(), Btype(accum), bias_diff);
      top_diff += dim_;
      accum = true;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(BiasLayer);

}  // namespace caffe
