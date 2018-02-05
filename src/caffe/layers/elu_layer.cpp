#include <algorithm>
#include <vector>

#include "caffe/layers/elu_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ELULayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  const int count = bottom[0]->count();
  float alpha = this->layer_param_.elu_param().alpha();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Ftype(0))
        + alpha * (exp(std::min(bottom_data[i], Ftype(0))) - 1.F);
  }
}

template <typename Ftype, typename Btype>
void ELULayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    const Btype* bottom_data = bottom[0]->cpu_data<Btype>();
    const Btype* top_data = top[0]->cpu_data<Btype>();
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    const int count = bottom[0]->count();
    float alpha = this->layer_param_.elu_param().alpha();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + (alpha + top_data[i]) * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ELULayer);
#endif

INSTANTIATE_CLASS_FB(ELULayer);
REGISTER_LAYER_CLASS(ELU);

}  // namespace caffe
