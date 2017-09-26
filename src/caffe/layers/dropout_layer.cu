#include <vector>
#include <device_launch_parameters.h>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
__global__ void
DropoutForward(const int n, const Dtype* in, const unsigned int* mask, const unsigned int threshold,
    const float scale, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    //    out[index] = in[index] * (mask[index] > threshold) * scale;
    if (mask[index] > threshold)
      out[index] = Dtype(static_cast<float>(in[index]) * scale);
    else
      out[index] = 0.;
  }
}

template<typename Ftype, typename Btype>
void
DropoutLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    cudaStream_t stream = Caffe::thread_stream();
    unsigned int* mask = static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(count, mask);
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>
        (count, bottom_data, mask, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    caffe_copy<Ftype>(count, bottom_data, top_data);
  }
}

template<typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff, const unsigned int* mask,
    const unsigned int threshold, const float scale, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    //    out_diff[index] = in_diff[index] * (mask[index] > threshold) * scale;
    if (mask[index] > threshold)
      out_diff[index] = Dtype(static_cast<float>(in_diff[index]) * scale);
    else
      out_diff[index] = 0.;
  }
}

template<typename Ftype, typename Btype>
void DropoutLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();

  if (propagate_down[0]) {
    if (this->phase_ == TRAIN) {  // Needed for TEST
      cudaStream_t stream = Caffe::thread_stream();
      const unsigned int* mask = static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropoutBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>
          (count, top_diff, mask, uint_thres_, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(DropoutLayer);

}  // namespace caffe
