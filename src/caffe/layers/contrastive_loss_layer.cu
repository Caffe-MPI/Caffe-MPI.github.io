#include <algorithm>
#include <vector>

#include "caffe/layers/contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ContrastiveLossLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub<Ftype>(
      count,
      bottom[0]->gpu_data<Ftype>(),  // a
      bottom[1]->gpu_data<Ftype>(),  // b
      diff_.template mutable_gpu_data<Ftype>());  // a_i-b_i
  caffe_gpu_powx<Ftype>(
      count,
      diff_.template mutable_gpu_data<Ftype>(),  // a_i-b_i
      Ftype(2),
      diff_sq_.template mutable_gpu_data<Ftype>());  // (a_i-b_i)^2
  caffe_gpu_gemv<Ftype>(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Ftype(1.0),
      diff_sq_.template gpu_data<Ftype>(),  // (a_i-b_i)^2
      summer_vec_.template gpu_data<Ftype>(),
      Ftype(0.0),
      dist_sq_.template mutable_gpu_data<Ftype>());  // \Sum (a_i-b_i)^2
  float margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  float loss = 0.F;
  for (int i = 0; i < bottom[0]->num(); ++i) {
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

template <typename Dtype>
__global__ void CLLBackward(const int count, const int channels,
    const float margin, const bool legacy_version, const float alpha,
    const Dtype* y, const Dtype* diff, const Dtype* dist_sq,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (static_cast<int>(y[n])) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
      float mdist(0.0);
      float beta(0.0);
      if (legacy_version) {
        mdist = (margin - dist_sq[n]);
        beta = -alpha;
      } else {
        float dist = sqrt(dist_sq[n]);
        mdist = (margin - dist);
        beta = -alpha * mdist / (dist + 1e-4F) * diff[i];
      }
      if (mdist > 0.0) {
        bottom_diff[i] = beta;
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}

template <typename Ftype, typename Btype>
void ContrastiveLossLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      float margin = this->layer_param_.contrastive_loss_param().margin();
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
      const float sign = (i == 0) ? 1 : -1;
      const float alpha = sign * top[0]->cpu_diff<Btype>()[0] /
          bottom[0]->num();
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLBackward<Btype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
          Caffe::thread_stream()>>>(
          count, channels, margin, legacy_version, alpha,
          bottom[2]->gpu_data<Btype>(),  // pair similarity 0 or 1
          diff_.template gpu_data<Btype>(),  // the cached eltwise difference between a and b
          dist_sq_.template gpu_data<Btype>(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff<Btype>());
      CUDA_POST_KERNEL_CHECK;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(ContrastiveLossLayer);

}  // namespace caffe
