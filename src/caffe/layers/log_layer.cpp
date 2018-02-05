#include <vector>

#include "caffe/layers/log_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void LogLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  NeuronLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  const float base = this->layer_param_.log_param().base();
  if (base != -1.F) {
    CHECK_GT(base, 0) << "base must be strictly positive.";
  }
  // If base == -1, interpret the base as e and set log_base = 1 exactly.
  // Otherwise, calculate its log explicitly.
  const float log_base = (base == -1.F) ? 1.F : log(base);
  CHECK(!isnan(log_base))
      << "NaN result: log(base) = log(" << base << ") = " << log_base;
  CHECK(!isinf(log_base))
      << "Inf result: log(base) = log(" << base << ") = " << log_base;
  base_scale_ = 1.F / log_base;
  CHECK(!isnan(base_scale_))
      << "NaN result: 1/log(base) = 1/log(" << base << ") = " << base_scale_;
  CHECK(!isinf(base_scale_))
      << "Inf result: 1/log(base) = 1/log(" << base << ") = " << base_scale_;
  input_scale_ = this->layer_param_.log_param().scale();
  input_shift_ = this->layer_param_.log_param().shift();
  backward_num_scale_ = input_scale_ / log_base;
}

template <typename Ftype, typename Btype>
void LogLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const int count = bottom[0]->count();
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  if (input_scale_ == 1.F && input_shift_ == 0.F) {
    caffe_log(count, bottom_data, top_data);
  } else {
    caffe_copy(count, bottom_data, top_data);
    if (input_scale_ != 1.F) {
      caffe_scal(count, Ftype(input_scale_), top_data);
    }
    if (input_shift_ != 0.F) {
      caffe_add_scalar(count, Ftype(input_shift_), top_data);
    }
    caffe_log(count, top_data, top_data);
  }
  if (base_scale_ != 1.F) {
    caffe_scal(count, Ftype(base_scale_), top_data);
  }
}

template <typename Ftype, typename Btype>
void LogLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) { return; }
  const int count = bottom[0]->count();
  const Btype* bottom_data = bottom[0]->cpu_data<Btype>();
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
  caffe_copy(count, bottom_data, bottom_diff);
  if (input_scale_ != 1.F) {
    caffe_scal(count, Btype(input_scale_), bottom_diff);
  }
  if (input_shift_ != 0.F) {
    caffe_add_scalar(count, Btype(input_shift_), bottom_diff);
  }
  caffe_powx(count, bottom_diff, Btype(-1), bottom_diff);
  if (backward_num_scale_ != 1.F) {
    caffe_scal(count, Btype(backward_num_scale_), bottom_diff);
  }
  caffe_mul(count, top_diff, bottom_diff, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(LogLayer);
#endif

INSTANTIATE_CLASS_FB(LogLayer);
REGISTER_LAYER_CLASS(Log);

}  // namespace caffe
