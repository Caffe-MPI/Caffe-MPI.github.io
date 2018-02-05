#include <vector>

#include "caffe/layers/threshold_layer.hpp"

namespace caffe {

template <typename Ftype>
__global__ void ThresholdForward(const int n, const float threshold,
    const Ftype* in, Ftype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? 1 : 0;
  }
}

template <typename Ftype, typename Btype>
void ThresholdLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ThresholdForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
      count, threshold_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FORWARD_ONLY_FB(ThresholdLayer);

}  // namespace caffe
