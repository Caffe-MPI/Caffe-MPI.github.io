#ifndef CAFFE_CUDNN_POOLING_LAYER_HPP_
#define CAFFE_CUDNN_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/**
 * @brief cuDNN implementation of PoolingLayer.
 *        Fallback to PoolingLayer for CPU mode.
 */
template <typename Ftype, typename Btype>
class CuDNNPoolingLayer : public PoolingLayer<Ftype, Btype> {
 public:
  explicit CuDNNPoolingLayer(const LayerParameter& param)
      : PoolingLayer<Ftype, Btype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual ~CuDNNPoolingLayer();
  // Currently, cuDNN does not support the extra top blob.
  virtual inline int MinTopBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  bool handles_setup_;
  cudnnHandle_t             handle_;
  cudnnTensorDescriptor_t   fwd_bottom_desc_, fwd_top_desc_;
  cudnnTensorDescriptor_t   bwd_bottom_desc_, bwd_top_desc_;
  cudnnPoolingDescriptor_t  pooling_desc_;
  cudnnPoolingMode_t        mode_;

  // private top blobs to workaround the inplace max pooling issue
  // for more, see: https://github.com/BVLC/caffe/issues/2015
  //                https://github.com/BVLC/caffe/issues/2688
  //                https://github.com/BVLC/caffe/pull/3574
  vector<shared_ptr<Blob> > private_top_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_POOLING_LAYER_HPP_
