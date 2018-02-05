#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/common.hpp"

namespace caffe {

template<typename Dtype>
void AdaDeltaSolver<Dtype>::AdaDeltaPreSolve() {
  // Add the extra history entries for AdaDelta after those from
  // SGDSolver::PreSolve
  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    this->history_.emplace_back(boost::make_shared<TBlob<Dtype>>(net_params[i]->shape()));
  }
}

#ifndef CPU_ONLY
template<typename Gtype, typename Wtype>
void
adadelta_reg_update_and_clear_gpu(int N,
    Gtype* g, Wtype* w, Wtype* h, Wtype* h2,
    float momentum, float delta, float local_rate, const std::string& regularization_type,
    float local_decay, void* handle, bool clear_grads);
#endif

template <typename Dtype>
void AdaDeltaSolver<Dtype>::ComputeUpdateValue(int param_id, void* handle, float rate,
    bool clear_grads) {
  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  shared_ptr<Blob> param = net_params[param_id];
  shared_ptr<TBlob<Dtype>> history = this->history_[param_id];
  shared_ptr<TBlob<Dtype>> update = this->update_[param_id];
  shared_ptr<TBlob<Dtype>> temp = this->temp_[param_id];
  const vector<float>& net_params_lr = this->net_->params_lr();
  float delta =  std::max(this->param_.delta(), 0.001F);
  float momentum = this->param_.momentum();
  float local_rate = rate * net_params_lr[param_id];
  size_t update_history_offset = net_params.size();
  if (Caffe::mode() == Caffe::CPU) {
    // compute square of gradient in update
    caffe_powx<Dtype>(param->count(), param->cpu_diff<Dtype>(), Dtype(2.F),
        update->mutable_cpu_data());

    // update history of gradients
    caffe_cpu_axpby<Dtype>(param->count(), Dtype(1.F - momentum), update->cpu_data(), momentum,
        history->mutable_cpu_data());

    // add delta to history to guard against dividing by zero later
    caffe_set<Dtype>(param->count(), delta, temp->mutable_cpu_data());

    caffe_add<Dtype>(param->count(), temp->cpu_data(),
        this->history_[update_history_offset + param_id]->cpu_data(), update->mutable_cpu_data());

    caffe_add<Dtype>(param->count(), temp->cpu_data(), history->cpu_data(),
        temp->mutable_cpu_data());

    // divide history of updates by history of gradients
    caffe_div<Dtype>(param->count(), update->cpu_data(), temp->cpu_data(),
        update->mutable_cpu_data());

    // jointly compute the RMS of both for update and gradient history
    caffe_powx<Dtype>(param->count(), update->cpu_data(), Dtype(0.5), update->mutable_cpu_data());

    // compute the update
    caffe_mul<Dtype>(param->count(), param->cpu_diff<Dtype>(), update->cpu_data(),
        param->mutable_cpu_diff<Dtype>());

    // compute square of update
    caffe_powx<Dtype>(param->count(), param->cpu_diff<Dtype>(), Dtype(2.F),
        update->mutable_cpu_data());

    // update history of updates
    caffe_cpu_axpby<Dtype>(param->count(), Dtype(1.F - momentum), update->cpu_data(), momentum,
        this->history_[update_history_offset + param_id]->mutable_cpu_data());

    // apply learning rate
    caffe_cpu_scale<Dtype>(param->count(), local_rate, param->cpu_diff<Dtype>(),
        param->mutable_cpu_diff<Dtype>());

    param->Update();
    if (clear_grads) {
      param->set_diff(0.F);
    }
  } else if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    const std::string& regularization_type = this->param_.regularization_type();
    const float decay = this->local_decay(param_id);
    const Type gtype = param->diff_type();
    if (gtype == tp<float16>()) {
      adadelta_reg_update_and_clear_gpu<float16, Dtype>(param->count(),
          param->mutable_gpu_diff<float16>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          this->history_[update_history_offset + param_id]->mutable_gpu_data(),
          momentum, delta, local_rate, regularization_type, decay, handle, clear_grads);
    } else if (gtype == tp<float>()) {
      adadelta_reg_update_and_clear_gpu<float, Dtype>(param->count(),
          param->mutable_gpu_diff<float>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          this->history_[update_history_offset + param_id]->mutable_gpu_data(),
          momentum, delta, local_rate, regularization_type, decay,  handle, clear_grads);
    } else if (gtype == tp<double>()) {
      adadelta_reg_update_and_clear_gpu<double, Dtype>(param->count(),
          param->mutable_gpu_diff<double>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          this->history_[update_history_offset + param_id]->mutable_gpu_data(),
          momentum, delta, local_rate, regularization_type, decay, handle, clear_grads);
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

INSTANTIATE_CLASS(AdaDeltaSolver);

REGISTER_SOLVER_CLASS(AdaDelta);

}  // namespace caffe
