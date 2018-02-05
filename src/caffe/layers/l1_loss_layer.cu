#include "caffe/layers/l1_loss_layer.hpp"

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void L1LossLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub<Ftype>(
      count,
      bottom[0]->gpu_data<Ftype>(),
      bottom[1]->gpu_data<Ftype>(),
      diff_.template mutable_gpu_data<Ftype>());
  caffe_gpu_sign<Ftype>(count, diff_.template gpu_data<Ftype>(),
      sign_.template mutable_gpu_data<Ftype>());
  Ftype abs_sum;
  caffe_gpu_asum(count, diff_.template gpu_data<Ftype>(), &abs_sum);
  Ftype loss = abs_sum / bottom[0]->num();
  top[0]->mutable_cpu_data<Ftype>()[0] = loss;
}

template <typename Ftype, typename Btype>
void L1LossLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Btype sign = (i == 0) ? 1 : -1;
      const Btype alpha = sign * top[0]->cpu_diff<Btype>()[0] / bottom[i]->num();
      caffe_gpu_axpby<Btype>(
          bottom[i]->count(),                     // count
          alpha,                                  // alpha
          sign_.template gpu_data<Btype>(),       // a
          Btype(0),                               // beta
          bottom[i]->mutable_gpu_diff<Btype>());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(L1LossLayer);

}  // namespace caffe
