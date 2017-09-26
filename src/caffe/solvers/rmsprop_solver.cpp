#include <algorithm>
#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

#ifndef CPU_ONLY
template<typename Gtype, typename Wtype>
void rmsprop_reg_update_and_clear_gpu(int N,
    Gtype* g, Wtype* w, Wtype* h,
    float rms_decay, float delta,  float local_rate, const std::string& regularization_type,
    float local_decay, void* handle, bool clear_grads);
#endif

template<typename Dtype>
void RMSPropSolver<Dtype>::ComputeUpdateValue(int param_id, void *handle, float rate,
    bool clear_grads) {
  shared_ptr<Blob> param = this->net_->learnable_params()[param_id];
  shared_ptr<TBlob<Dtype>> history = this->history_[param_id];
  shared_ptr<TBlob<Dtype>> update = this->update_[param_id];
  const vector<float>& net_params_lr = this->net_->params_lr();

  // get the learning rate
  float delta = std::max(this->param_.delta(), 0.0001F);
  float rms_decay = this->param_.rms_decay();
  float local_rate = rate * net_params_lr[param_id];

  if (Caffe::mode() == Caffe::CPU) {
    // compute square of gradient in update
    caffe_powx<Dtype>(param->count(), param->cpu_diff<Dtype>(), Dtype(2.F),
        update->mutable_cpu_data());

    // update history
    caffe_cpu_axpby<Dtype>(param->count(), Dtype(1.F - rms_decay), update->cpu_data(), rms_decay,
        history->mutable_cpu_data());

    // prepare update
    caffe_powx<Dtype>(param->count(), history->cpu_data(), Dtype(0.5), update->mutable_cpu_data());

    caffe_add_scalar<Dtype>(param->count(), delta, update->mutable_cpu_data());

    caffe_div<Dtype>(param->count(), param->cpu_diff<Dtype>(), update->cpu_data(),
        update->mutable_cpu_data());

    // scale and copy
    caffe_cpu_axpby<Dtype>(param->count(), local_rate, update->cpu_data(), Dtype(0.),
        param->mutable_cpu_diff<Dtype>());

    param->Update();
    if (clear_grads) {
      param->set_diff(0.F);
    }
  } else if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    const std::string& regularization_type = this->param_.regularization_type();
    float decay = this->local_decay(param_id);
    const Type gtype = param->diff_type();
    if (gtype == tp<float16>()) {
      rmsprop_reg_update_and_clear_gpu<float16, Dtype>(param->count(),
          param->mutable_gpu_diff<float16>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          rms_decay, delta, local_rate, regularization_type, decay, handle, clear_grads);
    } else if (gtype == tp<float>()) {
      rmsprop_reg_update_and_clear_gpu<float, Dtype>(param->count(),
          param->mutable_gpu_diff<float>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          rms_decay, delta, local_rate, regularization_type, decay, handle, clear_grads);
    } else if (gtype == tp<double>()) {
      rmsprop_reg_update_and_clear_gpu<double, Dtype>(param->count(),
          param->mutable_gpu_diff<double>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          rms_decay, delta, local_rate, regularization_type, decay, handle, clear_grads);
    } else {
      LOG(FATAL) << "Gradient type " << Type_Name(gtype) << " is not supported";
    }
#else
    NO_GPU;
#endif
  } else {
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(RMSPropSolver);

REGISTER_SOLVER_CLASS(RMSProp);

}  // namespace caffe
