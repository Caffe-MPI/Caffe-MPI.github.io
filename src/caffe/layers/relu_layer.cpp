#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ReLULayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  const int count = bottom[0]->count();
  float negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Ftype(0))
        + negative_slope * std::min(bottom_data[i], Ftype(0));
  }
}

template <typename Ftype, typename Btype>
void ReLULayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    const Btype* bottom_data = bottom[0]->cpu_data<Btype>();
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    const int count = bottom[0]->count();
    float negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS_FB(ReLULayer);

}  // namespace caffe
