// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropout_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void DropoutLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  NeuronLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Ftype, typename Btype>
void DropoutLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  NeuronLayer<Ftype, Btype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Ftype, typename Btype>
void DropoutLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli<float>(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Ftype, typename Btype>
void DropoutLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS_FB(DropoutLayer);

}  // namespace caffe
