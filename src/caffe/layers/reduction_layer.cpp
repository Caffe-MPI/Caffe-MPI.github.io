#include <vector>

#include "caffe/layers/reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ReductionLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  op_ = this->layer_param_.reduction_param().operation();
}

template <typename Ftype, typename Btype>
void ReductionLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.reduction_param().axis());
  // In the output, we'll keep all axes up to the reduction axis, but
  // throw away any after that.
  // Note: currently reducing along non-tail axes is not supported; otherwise,
  // we'd need to also copy any axes following an "end_axis".
  vector<int> top_shape(bottom[0]->shape().begin(),
                        bottom[0]->shape().begin() + axis_);
  top[0]->Reshape(top_shape);
  num_ = bottom[0]->count(0, axis_);
  dim_ = bottom[0]->count(axis_);
  CHECK_EQ(num_, top[0]->count());
  if (op_ == ReductionParameter_ReductionOp_SUM ||
      op_ == ReductionParameter_ReductionOp_MEAN) {
    vector<int> sum_mult_shape(1, dim_);
    sum_multiplier_.Reshape(sum_mult_shape);
    caffe_set<Ftype>(dim_, Ftype(1), sum_multiplier_.template mutable_cpu_data<Ftype>());
  }
  coeff_ = this->layer_param().reduction_param().coeff();
  if (op_ == ReductionParameter_ReductionOp_MEAN) {
    coeff_ /= dim_;
  }
}

template <typename Ftype, typename Btype>
void ReductionLayer<Ftype, Btype>::Forward_cpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  const Ftype* mult_data = NULL;
  if (sum_multiplier_.count() > 0) {
    mult_data = sum_multiplier_.template cpu_data<Ftype>();
  }
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  for (int i = 0; i < num_; ++i) {
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      *top_data = caffe_cpu_dot(dim_, mult_data, bottom_data);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      *top_data = caffe_cpu_asum(dim_, bottom_data);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      *top_data = caffe_cpu_dot(dim_, bottom_data, bottom_data);
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
    top_data = top[0]->mutable_cpu_data<Ftype>();
    caffe_scal(num_, Ftype(coeff_), top_data);
  }
}

template <typename Ftype, typename Btype>
void ReductionLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
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
    bottom_data = bottom[0]->cpu_data<Btype>();
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
  }
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
  for (int i = 0; i < num_; ++i) {
    const Btype bottom_coeff = (*top_diff) * coeff_;
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      caffe_set(dim_, bottom_coeff, bottom_diff);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      caffe_cpu_sign(dim_, bottom_data, bottom_diff);
      caffe_scal(dim_, bottom_coeff, bottom_diff);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      caffe_cpu_scale(dim_, Btype(2 * bottom_coeff), bottom_data, bottom_diff);
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

#ifdef CPU_ONLY
STUB_GPU(ReductionLayer);
#endif

INSTANTIATE_CLASS_FB(ReductionLayer);
REGISTER_LAYER_CLASS(Reduction);

}  // namespace caffe
