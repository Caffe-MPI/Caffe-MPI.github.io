#include <algorithm>
#include <vector>

#include "caffe/layers/bnll_layer.hpp"

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

template <typename Ftype>
__global__ void BNLLForward(const int n, const Ftype* in, Ftype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ?
        in[index] + log(1. + exp(-in[index])) :
        log(1. + exp(in[index]));
  }
}

template <typename Ftype, typename Btype>
void BNLLLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  BNLLForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

template <typename Btype>
__global__ void BNLLBackward(const int n, const Btype* in_diff,
    const Btype* in_data, Btype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    float expval = exp(min(in_data[index], Btype(kBNLL_THRESHOLD)));
    out_diff[index] = in_diff[index] * expval / (expval + 1.);
  }
}

template <typename Ftype, typename Btype>
void BNLLLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    BNLLBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(BNLLLayer);

}  // namespace caffe
