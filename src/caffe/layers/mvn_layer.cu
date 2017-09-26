#include <vector>

#include "caffe/layers/mvn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void MVNLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  // subtract mean
  caffe_gpu_gemv<Ftype>(CblasNoTrans, num, dim, Ftype(1. / dim), bottom_data,
      sum_multiplier_.template gpu_data<Ftype>(), Ftype(0.),
      mean_.template mutable_gpu_data<Ftype>());  // EX
  caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Ftype(-1.),
      mean_.template gpu_data<Ftype>(), sum_multiplier_.template gpu_data<Ftype>(), Ftype(0.),
      temp_.template mutable_gpu_data<Ftype>());
  caffe_gpu_add<Ftype>(temp_.count(), bottom_data, temp_.template gpu_data<Ftype>(),
      top_data);  // X-EX

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_powx<Ftype>(bottom[0]->count(), top_data, Ftype(2),
        temp_.template mutable_gpu_data<Ftype>());  // (X-EX)^2
    caffe_gpu_gemv<Ftype>(CblasNoTrans, num, dim, Ftype(1. / dim), temp_.template gpu_data<Ftype>(),
        sum_multiplier_.template gpu_data<Ftype>(), Ftype(0.),
        variance_.template mutable_gpu_data<Ftype>());  // E((X-EX)^2)

    // normalize variance
    caffe_gpu_powx<Ftype>(variance_.count(), variance_.template gpu_data<Ftype>(), Ftype(0.5),
        variance_.template mutable_gpu_data<Ftype>());

    caffe_gpu_add_scalar<Ftype>(variance_.count(), Ftype(eps_),
        variance_.template mutable_gpu_data<Ftype>());

    caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Ftype(1.),
        variance_.template gpu_data<Ftype>(), sum_multiplier_.template gpu_data<Ftype>(), Ftype(0.),
        temp_.template mutable_gpu_data<Ftype>());

    caffe_gpu_div<Ftype>(temp_.count(), top_data, temp_.template gpu_data<Ftype>(), top_data);
  }
}

template<typename Ftype, typename Btype>
void
MVNLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  const Btype* top_data = top[0]->gpu_data<Btype>();
  const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    caffe_gpu_mul<Btype>(temp_.count(), top_data, top_diff, bottom_diff);
    caffe_gpu_gemv<Btype>(CblasNoTrans, num, dim, Btype(1.), bottom_diff,
        sum_multiplier_.template gpu_data<Btype>(), Btype(0.),
        mean_.template mutable_gpu_data<Btype>());
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Btype(1.),
        mean_.template gpu_data<Btype>(), sum_multiplier_.template gpu_data<Btype>(), Btype(0.),
        bottom_diff);
    caffe_gpu_mul<Btype>(temp_.count(), top_data, bottom_diff, bottom_diff);

    caffe_gpu_gemv<Btype>(CblasNoTrans, num, dim, Btype(1.), top_diff,
        sum_multiplier_.template gpu_data<Btype>(), Btype(0.),
        mean_.template mutable_gpu_data<Btype>());
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Btype(1.),
        mean_.template gpu_data<Btype>(), sum_multiplier_.template gpu_data<Btype>(), Btype(1.),
        bottom_diff);

    caffe_gpu_axpby<Btype>(temp_.count(), Btype(1), top_diff, Btype(-1. / dim), bottom_diff);

    // put the squares of bottom into temp_
    caffe_gpu_powx<Btype>(temp_.count(), bottom_data, Btype(2),
        temp_.template mutable_gpu_data<Btype>());

    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Btype(1.),
        variance_.template gpu_data<Btype>(), sum_multiplier_.template gpu_data<Btype>(), Btype(0.),
        temp_.template mutable_gpu_data<Btype>());

    caffe_gpu_div<Btype>(temp_.count(), bottom_diff, temp_.template gpu_data<Btype>(), bottom_diff);
  } else {
    caffe_gpu_gemv<Btype>(CblasNoTrans, num, dim, Btype(1. / dim), top_diff,
        sum_multiplier_.template gpu_data<Btype>(), Btype(0.),
        mean_.template mutable_gpu_data<Btype>());
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Btype(-1.),
        mean_.template gpu_data<Btype>(), sum_multiplier_.template gpu_data<Btype>(), Btype(0.),
        temp_.template mutable_gpu_data<Btype>());
    caffe_gpu_add<Btype>(temp_.count(), top_diff, temp_.template gpu_data<Btype>(), bottom_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS_FB(MVNLayer);


}  // namespace caffe
