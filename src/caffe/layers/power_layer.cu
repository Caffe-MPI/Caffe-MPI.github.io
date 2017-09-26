#include <vector>

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void PowerLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == 0.F) {
    Ftype value = (power_ == 0) ? Ftype(1) : Ftype(pow(shift_, power_));
    caffe_gpu_set(count, value, top_data);
    return;
  }
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  caffe_copy(count, bottom_data, top_data);
  if (scale_ != 1.F) {
    caffe_gpu_scal(count, Ftype(scale_), top_data);
  }
  if (shift_ != 0.F) {
    caffe_gpu_add_scalar(count, Ftype(shift_), top_data);
  }
  if (power_ != 1.F) {
    caffe_gpu_powx(count, top_data, Ftype(power_), top_data);
  }
}

template <typename Ftype, typename Btype>
void PowerLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    const int count = bottom[0]->count();
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    if (diff_scale_ == 0.F || power_ == 1.F) {
      caffe_gpu_set(count, Btype(diff_scale_), bottom_diff);
    } else {
      const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
      // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
      //               = diff_scale * y / (shift + scale * x)
      if (power_ == 2.F) {
        // Special case for y = (shift + scale * x)^2
        //     -> dy/dx = 2 * scale * (shift + scale * x)
        //              = diff_scale * shift + diff_scale * scale * x
        caffe_gpu_axpby(count, Btype(diff_scale_ * scale_), bottom_data,
            Btype(0), bottom_diff);
        if (shift_ != 0.F) {
          caffe_gpu_add_scalar(count, Btype(diff_scale_ * shift_), bottom_diff);
        }
      } else if (shift_ == 0.F) {
        // Special case for y = (scale * x)^power
        //     -> dy/dx = scale * power * (scale * x)^(power - 1)
        //              = scale * power * (scale * x)^power * (scale * x)^(-1)
        //              = power * y / x
        const Btype* top_data = top[0]->gpu_data<Btype>();
        caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
        caffe_gpu_scal(count, Btype(power_), bottom_diff);
      } else {
        caffe_copy(count, bottom_data, bottom_diff);
        if (scale_ != 1.F) {
          caffe_gpu_scal(count, Btype(scale_), bottom_diff);
        }
        if (shift_ != 0.) {
          caffe_gpu_add_scalar(count, Btype(shift_), bottom_diff);
        }
        const Btype* top_data = top[0]->gpu_data<Btype>();
        caffe_gpu_div(count, top_data, bottom_diff, bottom_diff);
        if (diff_scale_ != 1.F) {
          caffe_gpu_scal(count, Btype(diff_scale_), bottom_diff);
        }
      }
    }
    caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(PowerLayer);

}  // namespace caffe
