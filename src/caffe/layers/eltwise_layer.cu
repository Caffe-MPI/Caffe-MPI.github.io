#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype>
__global__ void MaxForward(const int nthreads, const Ftype* bottom_data_a,
    const Ftype* bottom_data_b, const int blob_idx, Ftype* top_data,
    int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Ftype maxval = -max_dtype<Ftype>();
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

template <typename Ftype, typename Btype>
void EltwiseLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  int* mask = nullptr;
  const int count = top[0]->count();
  //  convert to Ftype
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  for (int i = 0; i < bottom.size(); ++i)
    bottom[i]->gpu_data<Ftype>();

  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data<Ftype>(),
        bottom[1]->gpu_data<Ftype>(), top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data<Ftype>(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set(count, Ftype(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, Ftype(coeffs_[i]), bottom[i]->gpu_data<Ftype>(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, bottom[0]->gpu_data<Ftype>(), bottom[1]->gpu_data<Ftype>(), 0, top_data, mask);
    for (int i = 2; i < bottom.size(); ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
          count, top_data, bottom[i]->gpu_data<Ftype>(), i-1, top_data, mask);
    }
    CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Btype>
__global__ void MaxBackward(const int nthreads, const Btype* top_diff,
    const int blob_idx, const int* mask, Btype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Btype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Ftype, typename Btype>
void EltwiseLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const int* mask = nullptr;
  const int count = top[0]->count();
  const Btype* top_data = top[0]->gpu_data<Btype>();
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Btype* bottom_data = bottom[i]->gpu_data<Btype>();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        {
          Btype* bottom_diff = bottom[i]->mutable_gpu_diff<Btype>();
          if (stable_prod_grad_) {
            bool initialized = false;
            for (int j = 0; j < bottom.size(); ++j) {
              if (i == j) { continue; }
              if (!initialized) {
                caffe_copy(count, bottom[j]->gpu_data<Btype>(), bottom_diff);
                initialized = true;
              } else {
                caffe_gpu_mul(count, bottom[j]->gpu_data<Btype>(), bottom_diff, bottom_diff);
              }
            }
          } else {
            caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
          }
          caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == 1.F) {
          bottom[i]->ShareDiff(*top[0]);
        } else {
          Btype* bottom_diff = bottom[i]->mutable_gpu_diff<Btype>();
          caffe_gpu_scale(count, Btype(coeffs_[i]), top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        {
          mask = max_idx_.gpu_data();
          Btype* bottom_diff = bottom[i]->mutable_gpu_diff<Btype>();
          MaxBackward<Btype>  // NOLINT_NEXT_LINE(whitespace/operators)
              <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>> (
              count, top_diff, i, mask, bottom_diff);
          CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
        }
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(EltwiseLayer);

}  // namespace caffe
