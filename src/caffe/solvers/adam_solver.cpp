#include <algorithm>
#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template<typename Dtype>
void AdamSolver<Dtype>::AdamPreSolve() {
  // Add the extra history entries for Adam after those from
  // SGDSolver::PreSolve
  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    this->history_.emplace_back(boost::make_shared<TBlob<Dtype>>(net_params[i]->shape()));
  }
}

#ifndef CPU_ONLY
template<typename Gtype, typename Wtype>
void adam_reg_update_and_clear_gpu(int N,
    Gtype* g, Wtype* w, Wtype* m, Wtype* v,
    float beta1, float beta2,  float eps_hat, float corrected_local_rate,
    const std::string& regularization_type, float local_decay,  void* handle, bool clear_grads);
#endif

template <typename Dtype>
void AdamSolver<Dtype>::ComputeUpdateValue(int param_id, void* handle, float rate,
    bool clear_grads) {
  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  shared_ptr<Blob> param = net_params[param_id];
  const vector<float>& net_params_lr = this->net_->params_lr();
  float local_rate = rate * net_params_lr[param_id];
  const float beta1 = this->param_.momentum();
  const float beta2 = this->param_.momentum2();

  // we create aliases for convenience
  size_t update_history_offset = net_params.size();
  TBlob<Dtype>* val_m = this->history_[param_id].get();
  TBlob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
  TBlob<Dtype>* val_t = this->temp_[param_id].get();

  const int t = this->iter_ + 1;
  const float correction = std::sqrt(1.F - pow(beta2, float(t))) / (1.F - pow(beta1, float(t)));
  const int N = param->count();
  const float eps_hat = std::max(this->param_.delta(), 0.0001F);

  if (Caffe::mode() == Caffe::CPU) {
    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
    caffe_cpu_axpby<Dtype>(N, Dtype(1.F - beta1), param->cpu_diff<Dtype>(), beta1,
        val_m->mutable_cpu_data());

    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
    caffe_mul<Dtype>(N, param->cpu_diff<Dtype>(), param->cpu_diff<Dtype>(),
        val_t->mutable_cpu_data());
    caffe_cpu_axpby<Dtype>(N, Dtype(1.F - beta2), val_t->cpu_data(), beta2,
        val_v->mutable_cpu_data());

    // set update
    caffe_powx<Dtype>(N, val_v->cpu_data(), Dtype(0.5), val_t->mutable_cpu_data());
    caffe_add_scalar<Dtype>(N, eps_hat, val_t->mutable_cpu_data());
    caffe_div<Dtype>(N, val_m->cpu_data(), val_t->cpu_data(), val_t->mutable_cpu_data());

    caffe_cpu_scale<Dtype>(N, Dtype(local_rate * correction), val_t->cpu_data(),
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
      adam_reg_update_and_clear_gpu<float16, Dtype>(N,
        param->mutable_gpu_diff<float16>(),
        param->mutable_gpu_data<Dtype>(),
          val_m->mutable_gpu_data(),
          val_v->mutable_gpu_data(),
          beta1, beta2, eps_hat, local_rate * correction, regularization_type,
          decay, handle, clear_grads);
    } else if (gtype == tp<float>()) {
      adam_reg_update_and_clear_gpu<float, Dtype>(N,
        param->mutable_gpu_diff<float>(),
          param->mutable_gpu_data<Dtype>(),
          val_m->mutable_gpu_data(),
          val_v->mutable_gpu_data(),
          beta1, beta2, eps_hat, local_rate * correction, regularization_type,
          decay, handle, clear_grads);
    } else if (gtype == tp<double>()) {
      adam_reg_update_and_clear_gpu<double, Dtype>(N,
        param->mutable_gpu_diff<double>(),
        param->mutable_gpu_data<Dtype>(),
        val_m->mutable_gpu_data(),
          val_v->mutable_gpu_data(),
          beta1, beta2, eps_hat, local_rate * correction, regularization_type,
          decay, handle, clear_grads);
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

INSTANTIATE_CLASS(AdamSolver);

REGISTER_SOLVER_CLASS(Adam);

}  // namespace caffe
