#include <vector>

#include "caffe/layers/absval_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void AbsValLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const int count = top[0]->count();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  caffe_gpu_abs(count, bottom[0]->gpu_data<Ftype>(), top_data);
}

template <typename Ftype, typename Btype>
void AbsValLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const int count = top[0]->count();
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  if (propagate_down[0]) {
    const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    caffe_gpu_sign(count, bottom_data, bottom_diff);
    caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(AbsValLayer);


}  // namespace caffe
