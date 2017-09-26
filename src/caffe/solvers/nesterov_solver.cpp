#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

#ifndef CPU_ONLY
template<typename Gtype, typename Wtype>
void nesterov_reg_update_and_clear_gpu(int N,
    Gtype* g, Wtype *w, Wtype* h,
    float momentum, float local_rate, const std::string& reg_type, float local_decay,
    void *handle, bool clear_grads);
#endif

template<typename Dtype>
void NesterovSolver<Dtype>::ComputeUpdateValue(int param_id, void *handle, float rate,
    bool clear_grads) {
  shared_ptr<Blob> param = this->net_->learnable_params()[param_id];
  shared_ptr<TBlob<Dtype>> history = this->history_[param_id];
  shared_ptr<TBlob<Dtype>> update = this->update_[param_id];
  const vector<float>& net_params_lr = this->net_->params_lr();
  float momentum = this->param_.momentum();
  float local_rate = rate * net_params_lr[param_id];
  if (Caffe::mode() == Caffe::CPU) {
    // save history momentum for stepping back
    caffe_copy<Dtype>(param->count(), history->cpu_data(), update->mutable_cpu_data());

    // update history
    caffe_cpu_axpby<Dtype>(param->count(), local_rate, param->cpu_diff<Dtype>(), momentum,
        history->mutable_cpu_data());

    // compute update: step back then over step
    caffe_cpu_axpby<Dtype>(param->count(), Dtype(1.F + momentum), history->cpu_data(), -momentum,
        update->mutable_cpu_data());

    // copy
    caffe_copy<Dtype>(param->count(), update->cpu_data(), param->mutable_cpu_diff<Dtype>());

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
      nesterov_reg_update_and_clear_gpu<float16, Dtype>(param->count(),
          param->mutable_gpu_diff<float16>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          momentum, local_rate, regularization_type, decay, handle, clear_grads);
    } else if (gtype == tp<float>()) {
      nesterov_reg_update_and_clear_gpu<float, Dtype>(param->count(),
          param->mutable_gpu_diff<float>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          momentum, local_rate, regularization_type, decay, handle, clear_grads);
    } else if (gtype == tp<double>()) {
      nesterov_reg_update_and_clear_gpu<double, Dtype>(param->count(),
          param->mutable_gpu_diff<double>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          momentum, local_rate, regularization_type, decay, handle, clear_grads);
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

INSTANTIATE_CLASS(NesterovSolver);

REGISTER_SOLVER_CLASS(Nesterov);

}  // namespace caffe
