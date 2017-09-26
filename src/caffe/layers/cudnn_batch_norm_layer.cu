#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNBatchNormLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {

  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  if (top[0] == bottom[0]) {
    top_data = private_top_->mutable_gpu_data<Ftype>();
  }

  double epsilon = this->eps_;

  const void* scale_data;
  const void* bias_data;
  void* global_mean;
  void* global_var;
  void* save_mean;
  void* save_inv_var;

  if (is_type<Ftype>(FLOAT16)) {
    if (this->phase_ == TRAIN) {
      global_mean = this->blobs_[0]->template mutable_gpu_data<float>();
      global_var  = this->blobs_[1]->template mutable_gpu_data<float>();
      save_mean    = save_mean_->template mutable_gpu_data<float>();
      save_inv_var = save_inv_var_->template mutable_gpu_data<float>();
    } else {
      global_mean = (void *) this->blobs_[0]->template gpu_data<float>();
      global_var  = (void *) this->blobs_[1]->template gpu_data<float>();
    }
    if (this->scale_bias_) {
      scale_data = this->blobs_[3]->template gpu_data<float>();
      bias_data  = this->blobs_[4]->template gpu_data<float>();
    } else {
      scale_data = scale_ones_->template gpu_data<float>();
      bias_data  = bias_zeros_->template gpu_data<float>();
    }
  } else {
    if (this->phase_ == TRAIN) {
      global_mean = this->blobs_[0]->template mutable_gpu_data<Ftype>();
      global_var  = this->blobs_[1]->template mutable_gpu_data<Ftype>();
      save_mean   = save_mean_->template mutable_gpu_data<Ftype>();
      save_inv_var = save_inv_var_->template mutable_gpu_data<Ftype>();
    } else {
      global_mean = (void *) this->blobs_[0]->template gpu_data<Ftype>();
      global_var  = (void *) this->blobs_[1]->template gpu_data<Ftype>();
    }
    if (this->scale_bias_) {
      scale_data = this->blobs_[3]->template gpu_data<Ftype>();
      bias_data  = this->blobs_[4]->template gpu_data<Ftype>();
    } else {
      scale_data = scale_ones_->template gpu_data<Ftype>();
      bias_data  = bias_zeros_->template gpu_data<Ftype>();
    }
  }
  if (this->phase_ == TRAIN) {
    double factor = 1. - this->moving_average_fraction_;
    if (this->iter() == 0) {
      factor = 1.0;
    }
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(Caffe::cudnn_handle(), mode_,
      cudnn::dataType<Ftype>::one, cudnn::dataType<Ftype>::zero,
        fwd_bottom_desc_, bottom_data, fwd_top_desc_, top_data,
        fwd_scale_bias_mean_var_desc_, scale_data, bias_data,
        factor, global_mean, global_var, epsilon, save_mean, save_inv_var));
  } else if (this->phase_ == TEST) {
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(Caffe::cudnn_handle(), mode_,
        cudnn::dataType<Ftype>::one, cudnn::dataType<Ftype>::zero,
        fwd_bottom_desc_, bottom_data, fwd_top_desc_, top_data,
        fwd_scale_bias_mean_var_desc_, scale_data, bias_data,
        global_mean, global_var, epsilon));
  } else {
    LOG(FATAL) << "Unknown phase";
  }
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));

  if (top[0] == bottom[0]) {
    private_bottom_->CopyDataFrom(*bottom[0]);
    top[0]->CopyDataFrom(*private_top_);
  }
}

template <typename Ftype, typename Btype>
void CuDNNBatchNormLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {

  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();

  if (top[0] == bottom[0]) {
    bottom_data = private_bottom_->gpu_data<Btype>();
  }
  double epsilon = this->eps_;
  const void* save_mean;
  const void* save_inv_var;
  const void* scale_data;
  void*  scale_diff;
  void*  bias_diff;

  if (is_type<Btype>(FLOAT16)) {
    save_mean    = save_mean_->template gpu_data<float>();
    save_inv_var = save_inv_var_->template gpu_data<float>();
    if (this->scale_bias_) {
      scale_data = this->blobs_[3]->template gpu_data<float>();
//      scale_diff = this->blobs_[3]->template mutable_gpu_diff<float>();
//      bias_diff  = this->blobs_[4]->template mutable_gpu_diff<float>();
// TODO: this is workaround required for bucket reduce
      scale_diff_tmp_.CopyDiffFrom(*this->blobs_[3]);
      bias_diff_tmp_.CopyDiffFrom(*this->blobs_[4]);
      scale_diff = scale_diff_tmp_.mutable_gpu_diff();
      bias_diff  = bias_diff_tmp_.mutable_gpu_diff();
    } else {
      scale_data = scale_ones_->template gpu_data<float>();
      scale_diff = scale_ones_->template mutable_gpu_diff<float>();
      bias_diff  = bias_zeros_->template mutable_gpu_diff<float>();
    }
  } else {
    save_mean = save_mean_->template gpu_data<Btype>();
    save_inv_var = save_inv_var_->template gpu_data<Btype>();
    if (this->scale_bias_) {
      scale_data = this->blobs_[3]->template gpu_data<Btype>();
      scale_diff = this->blobs_[3]->template mutable_gpu_diff<Btype>();
      bias_diff  = this->blobs_[4]->template mutable_gpu_diff<Btype>();
    } else {
      scale_data = scale_ones_->template gpu_data<Btype>();
      scale_diff = scale_ones_->template mutable_gpu_diff<Btype>();
      bias_diff  = bias_zeros_->template mutable_gpu_diff<Btype>();
    }
  }
  if (top[0] == bottom[0]) {
     // copy diff from top to private_top
     private_top_->CopyDiffFrom(*top[0]);
     top_diff = private_top_->gpu_diff<Btype>();
  }

  CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(), mode_,
      cudnn::dataType<Btype>::one, cudnn::dataType<Btype>::zero,
      cudnn::dataType<Btype>::one, cudnn::dataType<Btype>::one,
      bwd_bottom_desc_, bottom_data, bwd_bottom_desc_, top_diff, bwd_bottom_desc_, bottom_diff,
      bwd_scale_bias_mean_var_desc_, scale_data, scale_diff, bias_diff,
      epsilon, save_mean, save_inv_var));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));

  if (is_type<Btype>(FLOAT16)) {
    this->blobs_[3]->CopyDiffFrom(scale_diff_tmp_);
    this->blobs_[4]->CopyDiffFrom(bias_diff_tmp_);
  }
  }

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNBatchNormLayer);

}  // namespace caffe

#endif
