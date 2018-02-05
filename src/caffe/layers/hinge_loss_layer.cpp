#include <algorithm>
#include <vector>

#include "caffe/layers/hinge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void HingeLossLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* bottom_diff = bottom[0]->mutable_cpu_diff<Ftype>();
  const Ftype* label = bottom[1]->cpu_data<Ftype>();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
  }
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] =
          std::max(Ftype(0), Ftype(1. + bottom_diff[i * dim + j]));
    }
  }
  Ftype* loss = top[0]->mutable_cpu_data<Ftype>();
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1:
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Ftype, typename Btype>
void HingeLossLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    const Btype* label = bottom[1]->cpu_data<Btype>();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    for (int i = 0; i < num; ++i) {
      bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    }

    const Btype loss_weight = top[0]->cpu_diff<Btype>()[0];
    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, Btype(loss_weight / num), bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, Btype(loss_weight * 2 / num), bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS_FB(HingeLossLayer);
REGISTER_LAYER_CLASS(HingeLoss);

}  // namespace caffe
