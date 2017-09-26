#ifndef CAFFE_SPLIT_LAYER_HPP_
#define CAFFE_SPLIT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Creates a "split" path in the network by copying the bottom Blob
 *        into multiple top Blob%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Ftype, typename Btype>
class SplitLayer : public Layer<Ftype, Btype> {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  const char* type() const override { return "Split"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int MinTopBlobs() const override { return 1; }

 protected:
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Backward_cpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) override;
  void Backward_gpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) override;
  int count_;
};

}  // namespace caffe

#endif  // CAFFE_SPLIT_LAYER_HPP_
