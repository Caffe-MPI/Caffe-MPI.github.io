#ifndef CAFFE_BIAS_LAYER_HPP_
#define CAFFE_BIAS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes a sum of two input Blobs, with the shape of the
 *        latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        sum.
 *
 * The second input may be omitted, in which case it's learned as a parameter
 * of the layer.
 */
template <typename Ftype, typename Btype>
class BiasLayer : public Layer<Ftype, Btype> {
 public:
  explicit BiasLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Bias"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

 private:
  TBlob<Ftype> bias_multiplier_;
  int outer_dim_, bias_dim_, inner_dim_, dim_;
};



}  // namespace caffe

#endif  // CAFFE_BIAS_LAYER_HPP_
