#include <vector>

#include "caffe/layers/exp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ExpLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  NeuronLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  const float base = this->layer_param_.exp_param().base();
  if (base != -1.F) {
    CHECK_GT(base, 0) << "base must be strictly positive.";
  }
  // If base == -1, interpret the base as e and set log_base = 1 exactly.
  // Otherwise, calculate its log explicitly.
  const float log_base = base == -1.F ? 1.F : log(base);
  CHECK(!isnan(log_base))
      << "NaN result: log(base) = log(" << base << ") = " << log_base;
  CHECK(!isinf(log_base))
      << "Inf result: log(base) = log(" << base << ") = " << log_base;
  const float input_scale = this->layer_param_.exp_param().scale();
  const float input_shift = this->layer_param_.exp_param().shift();
  inner_scale_ = log_base * input_scale;
  outer_scale_ = input_shift == 0.F ? 1.F :
      (base != -1.F ? pow(base, input_shift) : exp(input_shift));
}

template <typename Ftype, typename Btype>
void ExpLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const int count = bottom[0]->count();
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  if (inner_scale_ == 1.F) {
    caffe_exp(count, bottom_data, top_data);
  } else {
    caffe_cpu_scale(count, Ftype(inner_scale_), bottom_data, top_data);
    caffe_exp(count, top_data, top_data);
  }
  if (outer_scale_ != 1.F) {
    caffe_scal(count, Ftype(outer_scale_), top_data);
  }
}

template <typename Ftype, typename Btype>
void ExpLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) { return; }
  const int count = bottom[0]->count();
  const Btype* top_data = top[0]->cpu_data<Btype>();
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
  caffe_mul(count, top_data, top_diff, bottom_diff);
  if (inner_scale_ != 1.F) {
    caffe_scal(count, Btype(inner_scale_), bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ExpLayer);
#endif

INSTANTIATE_CLASS_FB(ExpLayer);
REGISTER_LAYER_CLASS(Exp);

}  // namespace caffe
