#ifndef CAFFE_SILENCE_LAYER_HPP_
#define CAFFE_SILENCE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Ignores bottom blobs while producing no top blobs. (This is useful
 *        to suppress outputs during testing.)
 */
template <typename Ftype, typename Btype>
class SilenceLayer : public Layer<Ftype, Btype> {
 public:
  explicit SilenceLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {}

  virtual inline const char* type() const { return "Silence"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {}
  // We can't define Forward_gpu here, since STUB_GPU will provide
  // its own definition for CPU_ONLY mode.
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_SILENCE_LAYER_HPP_
