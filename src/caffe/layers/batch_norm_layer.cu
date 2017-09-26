#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void
BatchNormLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N * C);
  int top_size = top[0]->count();

  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const Ftype* global_mean = this->blobs_[0]->template gpu_data<Ftype>();
  const Ftype* global_var  = this->blobs_[1]->template gpu_data<Ftype>();

  if (this->phase_ == TEST) {
    //  Y = X- EX
    multicast_gpu<Ftype>(N, C, S, global_mean, temp_NCHW_->template mutable_gpu_data<Ftype>());
    caffe_gpu_sub<Ftype>(top_size, bottom_data, temp_NCHW_->template gpu_data<Ftype>(), top_data);
    //  inv_var = (eps + var)^(-0.5)
    caffe_copy<Ftype>(C, global_var, var_->template mutable_gpu_data<Ftype>());
    caffe_gpu_add_scalar<Ftype>(C, Ftype(eps_), var_->template mutable_gpu_data<Ftype>());
    caffe_gpu_powx<Ftype>(C, var_->template gpu_data<Ftype>(), Ftype(-0.5F),
        inv_var_->template mutable_gpu_data<Ftype>());
    //  X_norm = (X-EX) * inv_var
    multicast_gpu<Ftype>(N, C, S, inv_var_->template gpu_data<Ftype>(),
        temp_NCHW_->template mutable_gpu_data<Ftype>());
    caffe_gpu_mul<Ftype>(top_size, top_data, temp_NCHW_->template gpu_data<Ftype>(), top_data);
  } else {  // if (this->phase_ == TRAIN)
    // temp = EX
    compute_mean_per_channel_gpu<Ftype>(N, C, S, bottom_data,
        mean_->template mutable_gpu_data<Ftype>());
    multicast_gpu<Ftype>(N, C, S, mean_->template gpu_data<Ftype>(),
        temp_NCHW_->template mutable_gpu_data<Ftype>());
    // Y = X-EX
    caffe_gpu_sub<Ftype>(top_size, bottom_data, temp_NCHW_->template gpu_data<Ftype>(), top_data);
    // temp = (X-EX)^2;
    caffe_gpu_square<Ftype>(top_size, top[0]->gpu_data<Ftype>(),
        temp_NCHW_->template mutable_gpu_data<Ftype>());
    compute_mean_per_channel_gpu<Ftype>(N, C, S, temp_NCHW_->template gpu_data<Ftype>(),
        var_->template mutable_gpu_data<Ftype>());

    caffe_copy<Ftype>(C, var_->template gpu_data<Ftype>(),
        temp_C_->template mutable_gpu_data<Ftype>());
    //  temp= 1/sqrt(e + var(c)
    caffe_gpu_add_scalar<Ftype>(C, Ftype(eps_), temp_C_->template mutable_gpu_data<Ftype>());
    caffe_gpu_powx<Ftype>(C, temp_C_->template gpu_data<Ftype>(), Ftype(-0.5F),
        inv_var_->template mutable_gpu_data<Ftype>());
    multicast_gpu<Ftype>(N, C, S, inv_var_->template gpu_data<Ftype>(),
        temp_NCHW_->template mutable_gpu_data<Ftype>());
    // X_norm = (X-mean(c)) / sqrt(e + var(c))
    caffe_gpu_mul<Ftype>(top_size, top_data, temp_NCHW_->template gpu_data<Ftype>(), top_data);
    // copy x_norm for backward
    caffe_copy<Ftype>(top_size, top_data, x_norm_->template mutable_gpu_data<Ftype>());

    //  update global mean and variance
    if (iter_ > 1) {
      caffe_gpu_axpby<Ftype>(C, Ftype(1. - moving_average_fraction_),
          mean_->template gpu_data<Ftype>(), Ftype(moving_average_fraction_),
          this->blobs_[0]->template mutable_gpu_data<Ftype>());
      caffe_gpu_axpby<Ftype>(C, Ftype((1. - moving_average_fraction_)),
          var_->template gpu_data<Ftype>(), Ftype(moving_average_fraction_),
          this->blobs_[1]->template mutable_gpu_data<Ftype>());
    } else {
      caffe_copy<Ftype>(C, mean_->template gpu_data<Ftype>(),
          this->blobs_[0]->template mutable_gpu_data<Ftype>());
      caffe_copy<Ftype>(C, var_->template gpu_data<Ftype>(),
          this->blobs_[1]->template mutable_gpu_data<Ftype>());
    }
    iter_++;
  }

  //  -- STAGE 2:  Y = X_norm * scale[c] + shift[c]  -----------------
  if (scale_bias_) {
    //  Y = X_norm * scale[c]
    multicast_gpu<Ftype>(N, C, S, this->blobs_[3]->template gpu_data<Ftype>(),
        temp_NCHW_->template mutable_gpu_data<Ftype>());
    caffe_gpu_mul<Ftype>(top_size, top_data, temp_NCHW_->template gpu_data<Ftype>(), top_data);
    //  Y = Y + shift[c]
    multicast_gpu<Ftype>(N, C, S, this->blobs_[4]->template gpu_data<Ftype>(),
        temp_NCHW_->template mutable_gpu_data<Ftype>());
    caffe_gpu_add<Ftype>(top_size, top_data, temp_NCHW_->template mutable_gpu_data<Ftype>(),
        top_data);
  }
}


template<typename Ftype, typename Btype>
void BatchNormLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N * C);
  int top_size = top[0]->count();

  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  //  --  STAGE 1: compute dE/d(scale) and dE/d(shift) ---------------
  if (scale_bias_) {
    //  scale_diff: dE/d(scale)  =  sum(dE/dY .* X_norm)
    Btype* scale_diff = this->blobs_[3]->template mutable_gpu_diff<Btype>();
    caffe_gpu_mul<Btype>(top_size, top_diff, x_norm_->template gpu_data<Btype>(),
       temp_NCHW_->template mutable_gpu_diff<Btype>());
    compute_sum_per_channel_gpu(N, C, S, temp_NCHW_->template gpu_diff<Btype>(), scale_diff);
    //  shift_diff: dE/d(shift) = sum (dE/dY)
    Btype* shift_diff = this->blobs_[4]->template mutable_gpu_diff<Btype>();
    compute_sum_per_channel_gpu(N, C, S, top_diff, shift_diff);

    // --  STAGE 2: backprop dE/d(x_norm) = dE/dY .* scale ------------
    //  dE/d(X_norm) = dE/dY * scale[c]
    const Btype* scale_data = this->blobs_[3]->template gpu_data<Btype>();
    multicast_gpu<Btype>(N, C, S, scale_data, temp_NCHW_->template mutable_gpu_data<Btype>());
    caffe_gpu_mul<Btype>(top_size, top_diff, temp_NCHW_->template gpu_data<Btype>(),
        x_norm_->template mutable_gpu_diff<Btype>());

    top_diff = x_norm_->template gpu_diff<Btype>();
  }
  // --  STAGE 3: backprop dE/dY --> dE/dX --------------------------

  // ATTENTION: from now on we will use notation Y:= X_norm
  const Btype* top_data = x_norm_->template gpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();

  //  temp = mean(dE/dY .* Y)
  caffe_gpu_mul<Btype>(top_size, top_diff, top_data,
      temp_NCHW_->template mutable_gpu_diff<Btype>());
  compute_mean_per_channel_gpu<Btype>(N, C, S, temp_NCHW_->template gpu_diff<Btype>(),
      temp_C_->template mutable_gpu_diff<Btype>());
  multicast_gpu<Btype>(N, C, S, temp_C_->template gpu_diff<Btype>(),
      temp_NCHW_->template mutable_gpu_diff<Btype>());

  // bottom = mean(dE/dY .* Y) .* Y
  caffe_gpu_mul<Btype>(top_size, temp_NCHW_->template gpu_diff<Btype>(), top_data, bottom_diff);

  // temp = mean(dE/dY)
  compute_mean_per_channel_gpu<Btype>(N, C, S, top_diff,
      temp_C_->template mutable_gpu_diff<Btype>());
  multicast_gpu<Btype>(N, C, S, temp_C_->template gpu_diff<Btype>(),
      temp_NCHW_->template mutable_gpu_diff<Btype>());

  // bottom = mean(dE/dY) + mean(dE/dY .* Y) .* Y
  caffe_gpu_add<Btype>(top_size, temp_NCHW_->template gpu_diff<Btype>(), bottom_diff, bottom_diff);

  // bottom = dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_sub<Btype>(top_size, top_diff, bottom_diff, bottom_diff);

  // dE/dX = dE/dX ./ sqrt(var(X) + eps)
  multicast_gpu<Btype>(N, C, S, inv_var_->template gpu_data<Btype>(),
     temp_NCHW_->template mutable_gpu_data<Btype>());
  caffe_gpu_mul<Btype>(top_size, bottom_diff, temp_NCHW_->template gpu_data<Btype>(), bottom_diff);
}


INSTANTIATE_LAYER_GPU_FUNCS_FB(BatchNormLayer);

}  // namespace caffe
