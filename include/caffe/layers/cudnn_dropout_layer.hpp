#ifndef CAFFE_CUDNN_DROPOUT_LAYER_HPP_
#define CAFFE_CUDNN_DROPOUT_LAYER_HPP_

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/gpu_memory.hpp"

namespace caffe {

#ifdef USE_CUDNN
template <typename Ftype, typename Btype>
class CuDNNDropoutLayer : public DropoutLayer<Ftype, Btype> {
 public:
  explicit CuDNNDropoutLayer(const LayerParameter& param)
      : DropoutLayer<Ftype, Btype>(param), handles_setup_(false),
        bottom_desc_(nullptr), top_desc_(nullptr), dropout_desc_(nullptr),
        state_size_(0), reserve_space_size_(0) {
    seed_ = param.dropout_param().random_seed() >= 0 ?
        static_cast<uint64_t>(param.dropout_param().random_seed()) : Caffe::next_seed();
  }

  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual ~CuDNNDropoutLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  bool handles_setup_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  uint64_t seed_;
  size_t state_size_, reserve_space_size_;
  GPUMemory::Workspace states_, reserve_space_;
};

#endif  // USE_CUDNN

}  // namespace caffe

#endif  // CAFFE_CUDNN_DROPOUT_LAYER_HPP_
