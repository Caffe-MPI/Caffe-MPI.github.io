#include <algorithm>
#include <vector>

#include "caffe/layers/mvn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void MVNLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  mean_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  variance_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  if (this->layer_param_.mvn_param().across_channels()) {
    sum_multiplier_.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  } else {
    sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  }
  Ftype* multiplier_data = sum_multiplier_.template mutable_cpu_data<Ftype>();
  caffe_set(sum_multiplier_.count(), Ftype(1), multiplier_data);
  eps_ = std::max<Ftype>(this->layer_param_.mvn_param().eps(), min_dtype<Ftype>());
}

template<typename Ftype, typename Btype>
void MVNLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  // subtract mean
  caffe_cpu_gemv<Ftype>(CblasNoTrans, num, dim, Ftype(1. / dim), bottom_data,
      sum_multiplier_.template cpu_data<Ftype>(), Ftype(0.),
      mean_.template mutable_cpu_data<Ftype>());  // EX
  caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Ftype(-1.),
      mean_.template cpu_data<Ftype>(), sum_multiplier_.template cpu_data<Ftype>(), Ftype(0.),
      temp_.template mutable_cpu_data<Ftype>());
  caffe_add<Ftype>(temp_.count(), bottom_data, temp_.template cpu_data<Ftype>(), top_data);  // X-EX

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx<Ftype>(bottom[0]->count(), top_data, Ftype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Ftype>(CblasNoTrans, num, dim, Ftype(1. / dim), temp_.cpu_data(),
        sum_multiplier_.template cpu_data<Ftype>(), Ftype(0.),
        variance_.template mutable_cpu_data<Ftype>());  // E((X-EX)^2)

    // normalize variance
    caffe_powx<Ftype>(variance_.count(), variance_.template cpu_data<Ftype>(), Ftype(0.5),
        variance_.template mutable_cpu_data<Ftype>());

    caffe_add_scalar<Ftype>(variance_.count(), Ftype(eps_),
        variance_.template mutable_cpu_data<Ftype>());

    caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Ftype(1.),
        variance_.template cpu_data<Ftype>(), sum_multiplier_.template cpu_data<Ftype>(), Ftype(0.),
        temp_.template mutable_cpu_data<Ftype>());

    caffe_div(temp_.count(), top_data, temp_.template cpu_data<Ftype>(), top_data);
  }
}

template<typename Ftype, typename Btype>
void
MVNLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  const Btype* top_data = top[0]->cpu_data<Btype>();
  const Btype* bottom_data = bottom[0]->cpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
    caffe_cpu_gemv<Btype>(CblasNoTrans, num, dim, Btype(1.), bottom_diff,
        sum_multiplier_.template cpu_data<Btype>(), Btype(0.),
        mean_.template mutable_cpu_data<Btype>());
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Btype(1.),
        mean_.template cpu_data<Btype>(), sum_multiplier_.template cpu_data<Btype>(), Btype(0.),
        bottom_diff);
    caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

    caffe_cpu_gemv<Btype>(CblasNoTrans, num, dim, Btype(1.), top_diff,
        sum_multiplier_.template cpu_data<Btype>(), Btype(0.),
        mean_.template mutable_cpu_data<Btype>());
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Btype(1.),
        mean_.template cpu_data<Btype>(), sum_multiplier_.template cpu_data<Btype>(), Btype(1.),
        bottom_diff);

    caffe_cpu_axpby(temp_.count(), Btype(1), top_diff, Btype(-1. / dim), bottom_diff);

    // put the squares of bottom into temp_
    caffe_powx<Btype>(temp_.count(), bottom_data, Btype(2),
        temp_.template mutable_cpu_data<Btype>());
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Btype(1.),
        variance_.template cpu_data<Btype>(), sum_multiplier_.template cpu_data<Btype>(), Btype(0.),
        temp_.template mutable_cpu_data<Btype>());

    caffe_div<Btype>(temp_.count(), bottom_diff, temp_.template cpu_data<Btype>(), bottom_diff);
  } else {
    caffe_cpu_gemv<Btype>(CblasNoTrans, num, dim, Btype(1. / dim), top_diff,
        sum_multiplier_.template cpu_data<Btype>(), Btype(0.),
        mean_.template mutable_cpu_data<Btype>());
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Btype(-1.),
        mean_.template cpu_data<Btype>(), sum_multiplier_.template cpu_data<Btype>(), Btype(0.),
        temp_.template mutable_cpu_data<Btype>());
    caffe_add<Btype>(temp_.count(), top_diff, temp_.template cpu_data<Btype>(), bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(MVNLayer);
#endif

INSTANTIATE_CLASS_FB(MVNLayer);

REGISTER_LAYER_CLASS(MVN);

}  // namespace caffe
