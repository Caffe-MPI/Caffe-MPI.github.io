#ifndef CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_
#define CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/type.hpp"


namespace caffe {

#ifdef USE_CUDNN
template <typename Ftype, typename Btype>
class CuDNNBatchNormLayer : public BatchNormLayer<Ftype, Btype> {
 public:
  explicit CuDNNBatchNormLayer(const LayerParameter& param)
      : BatchNormLayer<Ftype, Btype>(param), handles_setup_(false),
        save_mean_(Blob::create(tp<Ftype>(), tp<Ftype>())),
        save_inv_var_(Blob::create(tp<Ftype>(), tp<Ftype>())) {
  }
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual ~CuDNNBatchNormLayer();

  bool skip_apply_update(int blob_id) const override {
    return blob_id < 3;
  }

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  // cuDNN descriptors / handles
  cudnnTensorDescriptor_t fwd_bottom_desc_, fwd_top_desc_;
  cudnnTensorDescriptor_t bwd_bottom_desc_, bwd_top_desc_;
  cudnnTensorDescriptor_t fwd_scale_bias_mean_var_desc_;
  cudnnTensorDescriptor_t bwd_scale_bias_mean_var_desc_;
  cudnnBatchNormMode_t mode_;

  bool handles_setup_;

  shared_ptr<Blob> save_mean_;
  shared_ptr<Blob> save_inv_var_;

  shared_ptr<Blob> scale_ones_;
  shared_ptr<Blob> bias_zeros_;

  shared_ptr<Blob> private_top_;
  shared_ptr<Blob> private_bottom_;

  TBlob<float> scale_diff_tmp_;
  TBlob<float> bias_diff_tmp_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_
