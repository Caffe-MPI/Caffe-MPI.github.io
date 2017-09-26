#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Ftype>
__global__ void PReLUForward(const int n, const int channels, const int dim,
    const Ftype* in, Ftype* out, const Ftype* slope_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

// CUDA kernel for bottom backward
template <typename Btype>
__global__ void PReLUBackward(const int n, const int channels, const int dim,
    const Btype* in_diff, const Btype* in_data, Btype* out_diff,
    const Btype* slope_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * slope_data[c]);
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Btype>
__global__ void PReLUParamBackward(const int n,
    const int rows, const int rowPitch, const Btype* in_diff,
    const Btype* in_data, Btype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0);
    for ( int k = 1; k < rows; k++ ) {
        out_diff[index] += in_diff[index + k*rowPitch]
           * in_data[index + k*rowPitch] * (in_data[index + k*rowPitch] <= 0);
    }
  }
}

template <typename Ftype, typename Btype>
void PReLULayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Ftype* slope_data = this->blobs_[0]->template gpu_data<Ftype>();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy<Ftype>(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  PReLUForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
      count, channels, dim, bottom_data, top_data, slope_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Ftype, typename Btype>
void PReLULayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.template gpu_data<Btype>();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Btype* slope_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
    int cdim = channels * dim;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUParamBackward<Btype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
      cdim, bottom[0]->num(), top[0]->offset(1), top_diff ,
      bottom_data ,
      backward_buff_.mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
    if (channel_shared_) {
      Btype dsum;
      caffe_gpu_dot(channels * dim, backward_buff_.gpu_diff(),
       multiplier_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), dsum, slope_diff);
    } else {
      caffe_gpu_gemv(CblasNoTrans, channels, dim, Btype(1.),
        backward_buff_.gpu_diff(), multiplier_.gpu_data(), Btype(1.),
        slope_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    const Btype* slope_data = this->blobs_[0]->template gpu_data<Btype>();
    int div_factor = channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUBackward<Btype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data,
        div_factor);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(PReLULayer);

}  // namespace caffe
