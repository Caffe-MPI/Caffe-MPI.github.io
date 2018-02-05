#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void AbsValLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  NeuronLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Ftype, typename Btype>
void AbsValLayer<Ftype, Btype>::Forward_cpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const int count = top[0]->count();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  caffe_abs(count, bottom[0]->cpu_data<Ftype>(), top_data);
}

template <typename Ftype, typename Btype>
void AbsValLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const int count = top[0]->count();
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  if (propagate_down[0]) {
    const Btype* bottom_data = bottom[0]->cpu_data<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    caffe_cpu_sign(count, bottom_data, bottom_diff);
    caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(AbsValLayer);
#endif

INSTANTIATE_CLASS_FB(AbsValLayer);
REGISTER_LAYER_CLASS(AbsVal);

}  // namespace caffe
