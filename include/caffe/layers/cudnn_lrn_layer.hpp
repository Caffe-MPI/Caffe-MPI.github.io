#ifndef CAFFE_CUDNN_LRN_LAYER_HPP_
#define CAFFE_CUDNN_LRN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/lrn_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
template <typename Ftype, typename Btype>
class CuDNNLRNLayer : public LRNLayer<Ftype, Btype> {
 public:
  explicit CuDNNLRNLayer(const LayerParameter& param)
      : LRNLayer<Ftype, Btype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual ~CuDNNLRNLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  bool handles_setup_;
  cudnnLRNDescriptor_t norm_desc_;
  cudnnTensorDescriptor_t fwd_bottom_desc_, fwd_top_desc_;
  cudnnTensorDescriptor_t bwd_bottom_desc_, bwd_top_desc_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_LRN_LAYER_HPP_
