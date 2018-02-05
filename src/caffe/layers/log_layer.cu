#include <vector>

#include "caffe/layers/log_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void LogLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const int count = bottom[0]->count();
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  if (input_scale_ == 1.F && input_shift_ == 0.F) {
    caffe_gpu_log(count, bottom_data, top_data);
  } else {
    caffe_copy(count, bottom_data, top_data);
    if (input_scale_ != 1.F) {
      caffe_gpu_scal(count, Ftype(input_scale_), top_data);
    }
    if (input_shift_ != 0.F) {
      caffe_gpu_add_scalar(count, Ftype(input_shift_), top_data);
    }
    caffe_gpu_log(count, top_data, top_data);
  }
  if (base_scale_ != 1.F) {
    caffe_gpu_scal(count, Ftype(base_scale_), top_data);
  }
}

template <typename Ftype, typename Btype>
void LogLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) { return; }
    const int count = bottom[0]->count();
    const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    caffe_copy(count, bottom_data, bottom_diff);
    if (input_scale_ != 1.F) {
      caffe_gpu_scal(count, Btype(input_scale_), bottom_diff);
    }
    if (input_shift_ != 0.F) {
      caffe_gpu_add_scalar(count, Btype(input_shift_), bottom_diff);
    }
    caffe_gpu_powx(count, bottom_diff, Btype(-1), bottom_diff);
    if (backward_num_scale_ != 1.F) {
      caffe_gpu_scal(count, Btype(backward_num_scale_), bottom_diff);
    }
    caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(LogLayer);

}  // namespace caffe
