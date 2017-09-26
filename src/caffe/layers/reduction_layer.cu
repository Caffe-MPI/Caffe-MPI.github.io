#include <vector>

#include "caffe/layers/reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ReductionLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  const Ftype* mult_data = NULL;
  if (sum_multiplier_.count() > 0) {
    mult_data = sum_multiplier_.template gpu_data<Ftype>();
  }
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  for (int i = 0; i < num_; ++i) {
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      caffe_gpu_dot(dim_, mult_data, bottom_data, top_data);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      caffe_gpu_asum(dim_, bottom_data, top_data);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      caffe_gpu_dot(dim_, bottom_data, bottom_data, top_data);
      break;
    default:
      LOG(FATAL) << "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
    }
    bottom_data += dim_;
    ++top_data;
  }
  if (coeff_ != 1.F) {
    // Reset the top_data pointer.
    top_data = top[0]->mutable_gpu_data<Ftype>();
    caffe_gpu_scal(num_, Ftype(coeff_), top_data);
  }
}

template <typename Ftype, typename Btype>
void ReductionLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) { return; }
  // Get bottom_data, if needed.
  const Btype* bottom_data = NULL;
  switch (op_) {
  // Operations that don't need bottom_data
  case ReductionParameter_ReductionOp_SUM:
  case ReductionParameter_ReductionOp_MEAN:
    break;
  // Operations that need bottom_data
  case ReductionParameter_ReductionOp_ASUM:
  case ReductionParameter_ReductionOp_SUMSQ:
    bottom_data = bottom[0]->gpu_data<Btype>();
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
  }
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
  for (int i = 0; i < num_; ++i) {
    const Btype bottom_coeff = (*top_diff) * coeff_;
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      caffe_gpu_set(dim_, bottom_coeff, bottom_diff);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      caffe_gpu_sign(dim_, bottom_data, bottom_diff);
      caffe_gpu_scal(dim_, bottom_coeff, bottom_diff);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      caffe_gpu_scale(dim_, Btype(2. * bottom_coeff), bottom_data, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
    }
    bottom_data += dim_;
    bottom_diff += dim_;
    ++top_diff;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(ReductionLayer);

}  // namespace caffe
