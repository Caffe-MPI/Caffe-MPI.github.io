#ifndef CAFFE_SOFTMAX_LAYER_HPP_
#define CAFFE_SOFTMAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Ftype, typename Btype>
class SoftmaxLayer : public Layer<Ftype, Btype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Softmax"; }
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

  int outer_num_;
  int inner_num_;
  int softmax_axis_;
  /// sum_multiplier is used to carry out sum using BLAS
  TBlob<Ftype> sum_multiplier_;
  /// scale is an intermediate TBlob to hold temporary results.
  TBlob<Ftype> scale_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_LAYER_HPP_
