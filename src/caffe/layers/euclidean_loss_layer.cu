#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void EuclideanLossLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub<Ftype>(
      count,
      bottom[0]->gpu_data<Ftype>(),
      bottom[1]->gpu_data<Ftype>(),
      diff_.template mutable_gpu_data<Ftype>());
  Ftype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  float loss = dot / bottom[0]->num() / 2.F;
  top[0]->mutable_cpu_data<Ftype>()[0] = loss;
}

template <typename Ftype, typename Btype>
void EuclideanLossLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Btype sign = (i == 0) ? 1 : -1;
      const Btype alpha = sign * top[0]->cpu_diff<Btype>()[0] / bottom[i]->num();
      caffe_gpu_axpby<Btype>(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.template gpu_data<Btype>(),       // a
          Btype(0),                           // beta
          bottom[i]->mutable_gpu_diff<Btype>());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(EuclideanLossLayer);

}  // namespace caffe
