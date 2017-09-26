#include "caffe/solver.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
Layer<Ftype, Btype>::Layer(const LayerParameter& param) : LayerBase(param) {
  // Set phase and copy blobs (if there are any).
  phase_ = layer_param_.phase();
  debug_ = layer_param_.debug();

  // Data&Math types
  if (!layer_param_.has_forward_type()) {
    layer_param_.set_forward_type(tp<Ftype>());
  }
  if (!layer_param_.has_backward_type()) {
    layer_param_.set_backward_type(tp<Btype>());
  }
  if (!layer_param_.has_forward_math()) {
    layer_param_.set_forward_math(tpmax<Ftype, float>());
  }
  if (!layer_param_.has_backward_math()) {
    layer_param_.set_backward_math(tpmax<Btype, float>());
  }

  Type ftype = layer_param_.forward_type();
  Type ftype_t = tp<Ftype>();
  if (ftype != ftype_t) {
    DLOG(WARNING) << "Overriding LayerParameter's forward type " << Type_Name(ftype)
                  << " by template's " << Type_Name(ftype_t) << " for Layer '"
                  << layer_param_.name() << "' of type '" << layer_param_.type() << "'";
    layer_param_.set_forward_type(ftype_t);
  }
  Type btype = param.backward_type();
  Type btype_t = tp<Btype>();
  if (btype != btype_t) {
    DLOG(WARNING) << "Overriding LayerParameter's backward type " << Type_Name(btype)
                  << " by template's " << Type_Name(btype_t) << " for Layer '"
                  << layer_param_.name() << "' of type '" << layer_param_.type() << "'";
    layer_param_.set_backward_type(btype_t);
  }

  if (layer_param_.blobs_size() > 0) {
    blobs_.resize(layer_param_.blobs_size());
    for (int i = 0; i < layer_param_.blobs_size(); ++i) {
      blobs_[i] = Blob::create<Ftype>();
      blobs_[i]->FromProto(layer_param_.blobs(i));
    }
  }
}

void LayerBase::InitMutex() {
  forward_mutex_.reset(new std::mutex());
}

void LayerBase::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

void LayerBase::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

const Solver* LayerBase::parent_solver() const {
  return parent_net_ == nullptr ? nullptr : parent_net_->parent_solver();
}

// Iteration counter maintained by Solver
int LayerBase::iter() const {
  const Solver* psolver = parent_solver();
  return psolver == nullptr ? 0 : psolver->iter();
}

int LayerBase::relative_iter() const {
  const Solver* psolver = parent_solver();
  return psolver == nullptr ? 0 : psolver->relative_iter();
}

INSTANTIATE_CLASS_FB(Layer);

}  // namespace caffe
