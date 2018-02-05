#include <algorithm>
#include <vector>

#include "caffe/layers/contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ContrastiveLossLayer<Ftype, Btype>::LayerSetUp(
  const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Ftype(1);
}

template <typename Ftype, typename Btype>
void ContrastiveLossLayer<Ftype, Btype>::Forward_cpu(
    const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  int count = bottom[0]->count();
  caffe_sub<Ftype>(
      count,
      bottom[0]->cpu_data<Ftype>(),  // a
      bottom[1]->cpu_data<Ftype>(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  float margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  float loss = 0.F;
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
    if (static_cast<int>(bottom[2]->cpu_data<Ftype>()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      if (legacy_version) {
        loss += std::max<float>(margin - dist_sq_.cpu_data()[i], 0.F);
      } else {
        float dist = std::max<float>(margin - sqrt(dist_sq_.cpu_data()[i]), 0.F);
        loss += dist*dist;
      }
    }
  }
  loss = loss / bottom[0]->num() / 2.F;
  top[0]->mutable_cpu_data<Ftype>()[0] = loss;
}

template <typename Ftype, typename Btype>
void ContrastiveLossLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  float margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const float sign = (i == 0) ? 1 : -1;
      const float alpha = sign * top[0]->cpu_diff<Btype>()[0] /
          bottom[i]->num();
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Btype* bout = bottom[i]->mutable_cpu_diff<Btype>();
        if (static_cast<int>(bottom[2]->cpu_data<Btype>()[j])) {  // similar pairs
          caffe_cpu_axpby(
              channels,
              Btype(alpha),
              diff_.template cpu_data<Btype>() + (j*channels),
              Btype(0.0),
              bout + (j*channels));
        } else {  // dissimilar pairs
          float mdist(0.0);
          float beta(0.0);
          if (legacy_version) {
            mdist = margin - dist_sq_.cpu_data()[j];
            beta = -alpha;
          } else {
            float dist = sqrt(dist_sq_.cpu_data()[j]);
            mdist = margin - dist;
            beta = -alpha * mdist / (dist + 1e-4);
          }
          if (mdist > 0.F) {
            caffe_cpu_axpby(
                channels,
                Btype(beta),
                diff_.template cpu_data<Btype>() + (j*channels),
                Btype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Btype(0), bout + (j*channels));
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ContrastiveLossLayer);
#endif

INSTANTIATE_CLASS_FB(ContrastiveLossLayer);
REGISTER_LAYER_CLASS(ContrastiveLoss);

}  // namespace caffe
