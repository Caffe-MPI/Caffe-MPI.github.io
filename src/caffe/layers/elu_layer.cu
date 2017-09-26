#include <algorithm>
#include <vector>

#include "caffe/layers/elu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ELUForward(const int n, const Dtype* in, Dtype* out,
    float alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] :
        Dtype(alpha * (exp(in[index]) - 1.));
  }
}

template <typename Ftype, typename Btype>
void ELULayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const int count = bottom[0]->count();
  float alpha = this->layer_param_.elu_param().alpha();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ELUForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
      count, bottom_data, top_data, alpha);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ELUBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* in_data,
    Dtype* out_diff, float alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_data[index] > 0 ? in_diff[index] :
        Dtype(in_diff[index] * (out_data[index] + alpha));
  }
}

template <typename Ftype, typename Btype>
void ELULayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    const Btype* top_data = top[0]->gpu_data<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    const int count = bottom[0]->count();
    float alpha = this->layer_param_.elu_param().alpha();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ELUBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, top_diff, top_data, bottom_data, bottom_diff, alpha);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS_FB(ELULayer);


}  // namespace caffe
