#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1. / (1. + exp(-in[index]));
  }
}

template <typename Ftype, typename Btype>
void SigmoidLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void SigmoidBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (Dtype(1.) - sigmoid_x);
  }
}

template <typename Ftype, typename Btype>
void SigmoidLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    const Btype* top_data = top[0]->gpu_data<Btype>();
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, top_diff, top_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(SigmoidLayer);

}  // namespace caffe
