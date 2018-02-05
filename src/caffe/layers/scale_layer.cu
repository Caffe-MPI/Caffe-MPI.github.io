#include <cfloat>
#include <vector>

#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ScaleForward(const int n, const Dtype* in,
    const Dtype* scale, const int scale_dim, const int inner_dim,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}

template <typename Dtype>
__global__ void ScaleBiasForward(const int n, const Dtype* in,
    const Dtype* scale, const Dtype* bias,
    const int scale_dim, const int inner_dim, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
  }
}

template <typename Ftype, typename Btype>
void ScaleLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const int count = top[0]->count();
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  if (bottom[0] == top[0]) {
    // in-place computation; need to store bottom data before overwriting it.
    // Note that this is only necessary for Backward; we could skip this if not
    // doing Backward, but Caffe currently provides no way of knowing whether
    // we'll need to do Backward at the time of the Forward call.
    caffe_copy<Ftype>(bottom[0]->count(), bottom[0]->gpu_data<Ftype>(),
               temp_.template mutable_gpu_data<Ftype>());
  }
  const Ftype* scale_data =
      (bottom.size() > 1 ? bottom[1] : this->blobs_[0].get())->template gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  if (bias_layer_) {
    const Ftype* bias_data = this->blobs_[bias_param_id_]->template gpu_data<Ftype>();
    ScaleBiasForward  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, bottom_data, scale_data, bias_data, scale_dim_, inner_dim_,
        top_data);
  } else {
    ScaleForward  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, bottom_data, scale_data, scale_dim_, inner_dim_, top_data);
  }
}

template <typename Ftype, typename Btype>
void ScaleLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (bias_layer_ &&
      this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
    bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_);
  }
  const bool scale_param = (bottom.size() == 1);
  Blob* scale = scale_param ? this->blobs_[0].get() : bottom[1];
  if ((!scale_param && propagate_down[1]) ||
      (scale_param && this->param_propagate_down_[0])) {
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    const bool in_place = (bottom[0] == top[0]);
    const Btype* bottom_data = (in_place ? &temp_ : bottom[0])->template gpu_data<Btype>();
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scale diff, and we're done.
    // If we're computing in-place (and not doing eltwise computation), this
    // hack doesn't work and we store the product in temp_.
    const bool is_eltwise = (bottom[0]->count() == scale->count());
    Btype* product = (is_eltwise ? scale->mutable_gpu_diff<Btype>() :
        (in_place ? temp_.template mutable_gpu_data<Btype>() :
         bottom[0]->mutable_gpu_diff<Btype>()));
    caffe_gpu_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) {
      Btype* sum_result = NULL;
      if (inner_dim_ == 1) {
        sum_result = product;
      } else if (sum_result_.count() == 1) {
        const Btype* sum_mult = sum_multiplier_.gpu_data();
        Btype* scale_diff = scale->mutable_cpu_diff<Btype>();
        if (scale_param) {
          Btype result;
          caffe_gpu_dot(inner_dim_, product, sum_mult, &result);
          *scale_diff += result;
        } else {
          caffe_gpu_dot(inner_dim_, product, sum_mult, scale_diff);
        }
      } else {
        const Btype* sum_mult = sum_multiplier_.gpu_data();
        sum_result = (outer_dim_ == 1) ?
            scale->mutable_gpu_diff<Btype>() : sum_result_.mutable_gpu_data();
        caffe_gpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,
                       Btype(1), product, sum_mult, Btype(0), sum_result);
      }
      if (outer_dim_ != 1) {
        const Btype* sum_mult = sum_multiplier_.gpu_data();
        if (scale_dim_ == 1) {
          Btype* scale_diff = scale->mutable_cpu_diff<Btype>();
          if (scale_param) {
            Btype result;
            caffe_gpu_dot(outer_dim_, sum_mult, sum_result, &result);
            *scale_diff += result;
          } else {
            caffe_gpu_dot(outer_dim_, sum_mult, sum_result, scale_diff);
          }
        } else {
          Btype* scale_diff = scale->mutable_gpu_diff<Btype>();
          caffe_gpu_gemv(CblasTrans, outer_dim_, scale_dim_,
                         Btype(1), sum_result, sum_mult, Btype(scale_param),
                         scale_diff);
        }
      }
    }
  }
  if (propagate_down[0]) {
    const int count = top[0]->count();
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    const Btype* scale_data = scale->gpu_data<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    ScaleForward  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, top_diff, scale_data, scale_dim_, inner_dim_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(ScaleLayer);

}  // namespace caffe
