#ifndef CAFFE_L1_LOSS_LAYER_HPP_
#define CAFFE_L1_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/*
 * L1Loss
 */
template <typename Ftype, typename Btype>
class L1LossLayer : public LossLayer<Ftype, Btype> {
 public:
  explicit L1LossLayer(const LayerParameter& param)
      : LossLayer<Ftype, Btype>(param), diff_() {}
  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  const char* type() const override { return "L1Loss"; }
  /**
   * Unlike most loss layers, in the L1LossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  bool AllowForceBackward(const int bottom_index) const override {
    return true;
  }

 protected:
  /// @copydoc L1LossLayer
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) override;
  void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) override;

  TBlob<Ftype> diff_;
  TBlob<Ftype> sign_;
};

}  // namespace caffe

#endif  // CAFFE_L1_LOSS_LAYER_HPP_
