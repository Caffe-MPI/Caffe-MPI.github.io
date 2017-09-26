#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/multinomial_logistic_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void MultinomialLogisticLossLayer<Ftype, Btype>::Reshape(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype, Btype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Ftype, typename Btype>
void MultinomialLogisticLossLayer<Ftype, Btype>::Forward_cpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  const Ftype* bottom_label = bottom[1]->cpu_data<Ftype>();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  float loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    float prob = std::max(
        bottom_data[i * dim + label],
        tol<Ftype>(kLOG_THRESHOLD, min_dtype<Ftype>()));
    loss -= log(prob);
  }
  top[0]->mutable_cpu_data<Ftype>()[0] = loss / num;
}

template <typename Ftype, typename Btype>
void MultinomialLogisticLossLayer<Ftype, Btype>::Backward_cpu(
    const vector<Blob*>& top, const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Btype* bottom_data = bottom[0]->cpu_data<Btype>();
    const Btype* bottom_label = bottom[1]->cpu_data<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), Btype(0), bottom_diff);
    const float scale = - top[0]->cpu_diff<Btype>()[0] / num;
    for (int i = 0; i < num; ++i) {
      int label = static_cast<int>(bottom_label[i]);
      float prob = std::max(
          bottom_data[i * dim + label],
          tol<Btype>(kLOG_THRESHOLD, min_dtype<Btype>()));
      bottom_diff[i * dim + label] = scale / prob;
    }
  }
}

INSTANTIATE_CLASS_FB(MultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(MultinomialLogisticLoss);

}  // namespace caffe
