#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void SoftmaxLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Ftype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Ftype(1.F), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Ftype, typename Btype>
void SoftmaxLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  Ftype* scale_data = scale_.template mutable_cpu_data<Ftype>();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        Ftype(-1.F), sum_multiplier_.cpu_data(), scale_data, Ftype(1.F), top_data);
    // exponentiation
    caffe_exp(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv(CblasTrans, channels, inner_num_, Ftype(1.F),
        top_data, sum_multiplier_.cpu_data(), Ftype(0.F), scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

template <typename Ftype, typename Btype>
void SoftmaxLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  const Btype* top_data = top[0]->cpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
  Btype* scale_data = scale_.template mutable_cpu_data<Btype>();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        Btype(-1.F), sum_multiplier_.template cpu_data<Btype>(), scale_data,
        Btype(1.F), bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS_FB(SoftmaxLayer);

}  // namespace caffe
