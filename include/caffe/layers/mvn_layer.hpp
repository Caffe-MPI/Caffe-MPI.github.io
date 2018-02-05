#ifndef CAFFE_MVN_LAYER_HPP_
#define CAFFE_MVN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Ftype, typename Btype>
class MVNLayer : public Layer<Ftype, Btype> {
 public:
  explicit MVNLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "MVN"; }
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

  TBlob<Ftype> mean_, variance_, temp_;

  /// sum_multiplier is used to carry out sum using BLAS
  TBlob<Ftype> sum_multiplier_;
  float eps_;
};

}  // namespace caffe

#endif  // CAFFE_MVN_LAYER_HPP_
