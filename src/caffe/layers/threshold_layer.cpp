#include <vector>

#include "caffe/layers/threshold_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ThresholdLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  NeuronLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.threshold_param().threshold();
}

template <typename Ftype, typename Btype>
void ThresholdLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] > threshold_) ? Ftype(1) : Ftype(0);
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ThresholdLayer, Forward);
#endif

INSTANTIATE_CLASS_FB(ThresholdLayer);
REGISTER_LAYER_CLASS(Threshold);

}  // namespace caffe
