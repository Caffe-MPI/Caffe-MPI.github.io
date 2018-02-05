#ifndef CAFFE_CUDNN_LCN_LAYER_HPP_
#define CAFFE_CUDNN_LCN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#ifndef CPU_ONLY
#include "caffe/util/gpu_memory.hpp"
#endif

namespace caffe {

#ifdef USE_CUDNN
template <typename Ftype, typename Btype>
class CuDNNLCNLayer : public LRNLayer<Ftype, Btype> {
 public:
  explicit CuDNNLCNLayer(const LayerParameter& param)
    : LRNLayer<Ftype, Btype>(param), handles_setup_(false), tempDataSize_(0) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual ~CuDNNLCNLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  bool handles_setup_;
  cudnnLRNDescriptor_t norm_desc_;
  cudnnTensorDescriptor_t fwd_bottom_desc_, fwd_top_desc_;
  cudnnTensorDescriptor_t bwd_bottom_desc_, bwd_top_desc_;

  int size_, pre_pad_;
  float alpha_, beta_, k_;

  size_t tempDataSize_;
  GPUMemory::Workspace temp1_, temp2_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_LCN_LAYER_HPP_
