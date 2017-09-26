#include <algorithm>
#include <vector>

#include "caffe/layers/bnll_layer.hpp"

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

template <typename Ftype, typename Btype>
void BNLLLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] > 0 ?
        bottom_data[i] + log(1. + exp(-bottom_data[i])) :
        log(1. + exp(bottom_data[i]));
  }
}

template <typename Ftype, typename Btype>
void BNLLLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    const Btype* bottom_data = bottom[0]->cpu_data<Btype>();
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    const int count = bottom[0]->count();
    float expval;
    for (int i = 0; i < count; ++i) {
      expval = exp(std::min(bottom_data[i], Btype(kBNLL_THRESHOLD)));
      bottom_diff[i] = top_diff[i] * expval / (expval + 1.);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BNLLLayer);
#endif

INSTANTIATE_CLASS_FB(BNLLLayer);
REGISTER_LAYER_CLASS(BNLL);

}  // namespace caffe
