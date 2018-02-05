#include <vector>

#include "caffe/layers/exp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ExpLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const int count = bottom[0]->count();
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  if (inner_scale_ == 1.F) {
    caffe_gpu_exp(count, bottom_data, top_data);
  } else {
    caffe_gpu_scale(count, Ftype(inner_scale_), bottom_data, top_data);
    caffe_gpu_exp(count, top_data, top_data);
  }
  if (outer_scale_ != 1.F) {
    caffe_gpu_scal(count, Ftype(outer_scale_), top_data);
  }
}

template <typename Ftype, typename Btype>
void ExpLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) { return; }
  const int count = bottom[0]->count();
  const Btype* top_data = top[0]->gpu_data<Btype>();
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
  caffe_gpu_mul(count, top_data, top_diff, bottom_diff);
  if (inner_scale_ != 1.F) {
    caffe_gpu_scal(count, Btype(inner_scale_), bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(ExpLayer);

}  // namespace caffe
