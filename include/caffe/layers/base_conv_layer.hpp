#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Ftype, typename Btype>
class BaseConvolutionLayer : public Layer<Ftype, Btype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }
  bool bias_term() const override  { return bias_term_; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  template <typename Dtype>
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false) {
    const Dtype* col_buff = input;
    if (!is_1x1_) {
      if (!skip_im2col) {
        conv_im2col_cpu<Dtype>(input, col_buffer_.template mutable_cpu_data<Dtype>());
      }
      col_buff = col_buffer_.template cpu_data<Dtype>();
    }
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
          group_, conv_out_spatial_dim_, kernel_dim_,
          (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
          (Dtype)0., output + output_offset_ * g);
    }
  }

  template <typename Dtype>
  void forward_cpu_bias(Dtype* output, const Dtype* bias) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_output_,
        out_spatial_dim_, 1, (Dtype)1., bias,
        bias_multiplier_.template cpu_data<Dtype>(),
        (Dtype)1., output);
  }

  template <typename Dtype>
  void backward_cpu_gemm(const Dtype* output, const Dtype* weights,
      Dtype* input) {
    Dtype* col_buff = col_buffer_.template mutable_cpu_data<Dtype>();
    if (is_1x1_) {
      col_buff = input;
    }
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm(CblasTrans, CblasNoTrans, kernel_dim_,
          conv_out_spatial_dim_, conv_out_channels_ / group_,
          (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
          (Dtype)0., col_buff + col_offset_ * g);
    }
    if (!is_1x1_) {
      conv_col2im_cpu(col_buff, input);
    }
  }

  template <typename Dtype>
  void weight_cpu_gemm(const Dtype* input, const Dtype* output,
      Dtype* weights) {
    const Dtype* col_buff = input;
    if (!is_1x1_) {
      conv_im2col_cpu<Dtype>(input, col_buffer_.template mutable_cpu_data<Dtype>());
      col_buff = col_buffer_.template cpu_data<Dtype>();
    }
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
          kernel_dim_, conv_out_spatial_dim_,
          (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
          (Dtype)1., weights + weight_offset_ * g);
    }
  }

  template <typename Dtype>
  void backward_cpu_bias(Dtype* bias, const Dtype* input) {
    caffe_cpu_gemv(CblasNoTrans, num_output_, out_spatial_dim_, (Dtype)1.,
        input, bias_multiplier_.template cpu_data<Dtype>(), (Dtype)1., bias);
  }

#ifndef CPU_ONLY

  template <typename Dtype>
  void forward_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output,
      bool skip_im2col = false) {
    const Dtype* col_buff = input;
    if (!is_1x1_) {
      if (!skip_im2col) {
        conv_im2col_gpu<Dtype>(input, col_buffer_.template mutable_gpu_data<Dtype>());
      }
      col_buff = col_buffer_.template gpu_data<Dtype>();
    }
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
          group_, conv_out_spatial_dim_, kernel_dim_,
          (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
          (Dtype)0., output + output_offset_ * g);
    }
  }

  template <typename Dtype>
  void forward_gpu_bias(Dtype* output, const Dtype* bias) {
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_output_,
        out_spatial_dim_, 1, (Dtype)1., bias,
        bias_multiplier_.template gpu_data<Dtype>(),
        (Dtype)1., output);
  }

  template <typename Dtype>
  void backward_gpu_gemm(const Dtype* output, const Dtype* weights,
      Dtype* input) {
    Dtype* col_buff = col_buffer_.template mutable_gpu_data<Dtype>();
    if (is_1x1_) {
      col_buff = input;
    }
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm(CblasTrans, CblasNoTrans, kernel_dim_,
          conv_out_spatial_dim_, conv_out_channels_ / group_,
          (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
          (Dtype)0., col_buff + col_offset_ * g);
    }
    if (!is_1x1_) {
      conv_col2im_gpu(col_buff, input);
    }
  }

  template <typename Dtype>
  void weight_gpu_gemm(const Dtype* input, const Dtype* output,
      Dtype* weights) {
    const Dtype* col_buff = input;
    if (!is_1x1_) {
      conv_im2col_gpu<Dtype>(input, col_buffer_.template mutable_gpu_data<Dtype>());
      col_buff = col_buffer_.template gpu_data<Dtype>();
    }
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
          kernel_dim_, conv_out_spatial_dim_,
          (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
          (Dtype)1., weights + weight_offset_ * g);
    }
  }

  template <typename Dtype>
  void backward_gpu_bias(Dtype* bias, const Dtype* input) {
    caffe_gpu_gemv(CblasNoTrans, num_output_, out_spatial_dim_, (Dtype)1.,
        input, bias_multiplier_.template gpu_data<Dtype>(), (Dtype)1., bias);
  }

#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  TBlob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  TBlob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  TBlob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  TBlob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  TBlob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  template <typename Dtype>
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  template <typename Dtype>
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  template <typename Dtype>
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  template <typename Dtype>
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  TBlob<Ftype> col_buffer_;
  TBlob<Ftype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
