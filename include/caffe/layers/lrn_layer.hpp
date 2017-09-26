#ifndef CAFFE_LRN_LAYER_HPP_
#define CAFFE_LRN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/split_layer.hpp"

namespace caffe {

/**
 * @brief Normalize the input in a local region across or within feature maps.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Ftype, typename Btype>
class LRNLayer : public Layer<Ftype, Btype> {
 public:
  explicit LRNLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "LRN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  virtual void CrossChannelForward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void CrossChannelForward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void WithinChannelForward(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void CrossChannelBackward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void CrossChannelBackward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void WithinChannelBackward(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  int size_;
  int pre_pad_;
  float alpha_;
  float beta_;
  float k_;
  int num_;
  int channels_;
  int height_;
  int width_;

  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the intermediate summing results
  TBlob<Ftype> scale_;  // Conversions unavoidable if Ftype != Btype

  // Fields used for normalization WITHIN_CHANNEL
  shared_ptr<SplitLayer<Ftype, Btype> > split_layer_;
  vector<Blob*> split_top_vec_;
  shared_ptr<PowerLayer<Ftype, Btype> > square_layer_;
  TBlob<Ftype> square_input_;
  TBlob<Btype> square_output_;
  vector<Blob*> square_bottom_vec_;
  vector<Blob*> square_top_vec_;
  shared_ptr<PoolingLayer<Ftype, Btype> > pool_layer_;
  TBlob<Btype> pool_output_;
  vector<Blob*> pool_top_vec_;
  shared_ptr<PowerLayer<Ftype, Btype> > power_layer_;
  TBlob<Btype> power_output_;
  vector<Blob*> power_top_vec_;
  shared_ptr<EltwiseLayer<Ftype, Btype> > product_layer_;
  TBlob<Ftype> product_input_;
  vector<Blob*> product_bottom_vec_;
};

}  // namespace caffe

#endif  // CAFFE_LRN_LAYER_HPP_
