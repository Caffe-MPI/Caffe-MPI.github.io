#include "caffe/layers/l1_loss_layer.hpp"

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void L1LossLayer<Ftype, Btype>::Reshape(
  const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype, Btype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  sign_.ReshapeLike(*bottom[0]);
}


template <typename Ftype, typename Btype>
void L1LossLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
                                           const vector<Blob*>& top) {
  int count = bottom[0]->count();
  caffe_sub<Ftype>(
      count,
      bottom[0]->cpu_data<Ftype>(),
      bottom[1]->cpu_data<Ftype>(),
      diff_.template mutable_cpu_data<Ftype>());
  caffe_cpu_sign<Ftype>(count, diff_.template cpu_data<Ftype>(),
      sign_.template mutable_cpu_data<Ftype>());
  Ftype abs_sum = caffe_cpu_asum(count, diff_.template cpu_data<Ftype>());
  Ftype loss = abs_sum / bottom[0]->num();
  top[0]->mutable_cpu_data<Ftype>()[0] = loss;
}


template <typename Ftype, typename Btype>
void L1LossLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Btype sign = (i == 0) ? 1 : -1;
      const Btype alpha = sign * top[0]->cpu_diff<Btype>()[0] / bottom[i]->num();
      caffe_cpu_axpby<Btype>(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          sign_.template cpu_data<Btype>(),       // a
          Btype(0),                           // beta
          bottom[i]->mutable_cpu_diff<Btype>());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(L1LossLayer);
#endif

INSTANTIATE_CLASS_FB(L1LossLayer);
REGISTER_LAYER_CLASS(L1Loss);

}  // namespace caffe
