#ifndef CAFFE_DECONV_LAYER_HPP_
#define CAFFE_DECONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolve the input with a bank of learned filters, and (optionally)
 *        add biases, treating filters and convolution parameters in the
 *        opposite sense as ConvolutionLayer.
 *
 *   ConvolutionLayer computes each output value by dotting an input window with
 *   a filter; DeconvolutionLayer multiplies each input value by a filter
 *   elementwise, and sums over the resulting output windows. In other words,
 *   DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
 *   reversed. DeconvolutionLayer reuses ConvolutionParameter for its
 *   parameters, but they take the opposite sense as in ConvolutionLayer (so
 *   padding is removed from the output rather than added to the input, and
 *   stride results in upsampling rather than downsampling).
 */
template <typename Ftype, typename Btype>
class DeconvolutionLayer : public BaseConvolutionLayer<Ftype, Btype> {
 public:
  explicit DeconvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Ftype, Btype>(param) {}

  virtual inline const char* type() const { return "Deconvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual inline bool reverse_dimensions() { return true; }
  virtual void compute_output_shape();
};

}  // namespace caffe

#endif  // CAFFE_DECONV_LAYER_HPP_
