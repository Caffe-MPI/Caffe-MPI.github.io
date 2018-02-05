#ifndef CAFFE_BATCHNORM_LAYER_HPP_
#define CAFFE_BATCHNORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#define BN_VAR_CLIP_START 200
#define BN_VAR_CLIP_CONST 4.0

namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization described in [1].  For
 * each channel in the data (i.e. axis 1), it subtracts the mean and divides
 * by the variance, where both statistics are computed across both spatial
 * dimensions and across the different examples in the batch.
 *
 * By default, during training time, the network is computing global mean/
 * variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input.  You can manually
 * toggle whether the network is accumulating or using the statistics via the
 * use_global_stats option.  IMPORTANT: for this feature to work, you MUST
 * set the learning rate to zero for all three parameter blobs, i.e.,
 * param {lr_mult: 0} three times in the layer definition.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor.  It is possible (though a bit cumbersome) to implement
 * this in caffe using a single-channel DummyDataLayer filled with zeros,
 * followed by a Convolution layer with output the same size as the current.
 * This produces a channel-specific value that can be added or multiplied by
 * the BatchNorm layer's output.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Ftype, typename Btype>
class BatchNormLayer : public Layer<Ftype, Btype> {
 public:
  explicit BatchNormLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual inline const char* type() const { return "BatchNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
     const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  //  multicast x[c] into y[.,c,...]
  template <typename Dtype>
  void multicast_cpu(int N, int C, int S, const Dtype *x, Dtype *y ) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1, Dtype(1.),
        ones_N_->template cpu_data<Dtype>(), x, Dtype(0.),
        temp_NC_->template mutable_cpu_data<Dtype>());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S , 1,
        Dtype(1.), temp_NC_->template cpu_data<Dtype>(), ones_HW_->template cpu_data<Dtype>(),
        Dtype(0.), y);
  }

  //  y[c] = sum x(.,c,...)
  template <typename Dtype>
  void compute_sum_per_channel_cpu(int N, int C, int S, const Dtype *x, Dtype *y ) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, N * C, S, Dtype(1.), x,
        ones_HW_->template cpu_data<Dtype>(), Dtype(0.),
        temp_NC_->template mutable_cpu_data<Dtype>());
    caffe_cpu_gemv<Dtype>(CblasTrans, N, C , Dtype(1.), temp_NC_->template cpu_data<Dtype>(),
        ones_N_->template cpu_data<Dtype>(), Dtype(0.), y);
  }

  // y[c] = mean x(.,c,...)
  template <typename Dtype>
  void compute_mean_per_channel_cpu(int N, int C, int S, const Dtype *x, Dtype *y) {
    Dtype F = 1. / (N * S);
    compute_sum_per_channel_cpu(N, C, S, x, y);
    caffe_cpu_scale(C, F, y, y);
  }

#ifndef CPU_ONLY
  // multicast x[c] into y[.,c,...]
  template <typename Dtype>
  void multicast_gpu(int N, int C, int S, const Dtype *x, Dtype *y ) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1, Dtype(1.),
       ones_N_->template gpu_data<Dtype>(), x, Dtype(0.),
       temp_NC_-> template mutable_gpu_data<Dtype>());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S , 1, Dtype(1.),
       temp_NC_->gpu_data<Dtype>(), ones_HW_->template gpu_data<Dtype>(), Dtype(0.), y);
  }

  // y[c] = sum x(.,c,...)
  template <typename Dtype>
  void compute_sum_per_channel_gpu(int N, int C, int S, const Dtype *x, Dtype *y) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N * C, S, Dtype(1.), x,
        ones_HW_->template gpu_data<Dtype>(),
        Dtype(0.), temp_NC_-> template mutable_gpu_data<Dtype>());
    caffe_gpu_gemv<Dtype>(CblasTrans, N, C, Dtype(1.), temp_NC_->gpu_data<Dtype>(),
        ones_N_->template gpu_data<Dtype>(), Dtype(0.), y);
  }

  // y[c] = mean x(.,c,...)
  template <typename Dtype>
  void compute_mean_per_channel_gpu(int N, int C, int S, const Dtype *x, Dtype *y) {
    Dtype F = 1. / (N * S);
    compute_sum_per_channel_gpu(N, C, S, x, y);
    caffe_gpu_scal(C, F, y);
  }
#endif

  double moving_average_fraction_, eps_;
  int channels_, iter_;
  bool use_global_stats_, clip_variance_, scale_bias_;
  shared_ptr<Blob> mean_, var_, inv_var_, x_norm_;
  // auxiliary arrays used for sums and broadcast
  shared_ptr<Blob> ones_N_, ones_HW_, ones_C_, temp_C_, temp_NC_, temp_NCHW_;
};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
