#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <device_launch_parameters.h>

#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
__global__
void transform_kernel(int N, int C,
                      int H, int W,  // original size
                      int Hc, int Wc,  // cropped size
                      bool param_mirror,
                      int datum_height, int datum_width,  // offsets
                      int crop_size, Phase phase,
                      size_t sizeof_element,
                      const Dtype *in,
                      Dtype *out,  // buffers
                      float scale,
                      int has_mean_file,
                      int has_mean_values,
                      float *mean,
                      const unsigned int *random_numbers) {
  const int c = blockIdx.y;

  // loop over images
  for (int n = blockIdx.x; n < N; n += gridDim.x) {
    // get mirror and offsets
    unsigned int rand1 = random_numbers[n*3    ];
    unsigned int rand2 = random_numbers[n*3 + 1];
    unsigned int rand3 = random_numbers[n*3 + 2];

    bool mirror = param_mirror && (rand1 % 2);
    int h_off = 0, w_off = 0;
    if (crop_size) {
      if (phase == TRAIN) {
        h_off = rand2 % (datum_height - crop_size + 1);
        w_off = rand3 % (datum_width - crop_size + 1);
      } else {
        h_off = (datum_height - crop_size) / 2;
        w_off = (datum_width - crop_size) / 2;
      }
    }

    const uint8_t *in_ptri;
    const float *in_ptrf;
    // offsets into start of (image, channel) = (n, c)
    // channel is handled by blockIdx.y
    // Initial offset per Dtype:
    const Dtype *in_ptr  = &in[n*C*H*W];
    // Element-specific offset to a channel c
    if (sizeof_element == sizeof(uint8_t)) {
      in_ptri = reinterpret_cast<const uint8_t*>(in_ptr);
      in_ptri += c*H*W;
    } else if (sizeof_element == sizeof(float)) {
      in_ptrf = reinterpret_cast<const float*>(in_ptr);
      in_ptrf += c*H*W;
    } else {
      in_ptr += c*H*W;
    }

    Dtype *out_ptr = &out[n*C*Hc*Wc + c*Hc*Wc];
    Dtype element;
    // loop over pixels using threads
    for (int h = threadIdx.y; h < Hc; h += blockDim.y) {
      for (int w = threadIdx.x; w < Wc; w += blockDim.x) {
        // get the indices for in, out buffers
        int in_idx  = (h_off + h) * W + w_off + w;
        int out_idx = mirror ? h * Wc + (Wc - 1 - w) : h * Wc + w;

        if (sizeof_element == sizeof(uint8_t)) {
          element = in_ptri[in_idx];
        } else if (sizeof_element == sizeof(float)) {
          element = in_ptrf[in_idx];
        } else {
          element = in_ptr[in_idx];
        }

        // perform the transform
        if (has_mean_file) {
          out_ptr[out_idx] = (element - mean[c*H*W + in_idx]) * scale;
        } else {
          if (has_mean_values) {
            out_ptr[out_idx] = (element - mean[c]) * scale;
          } else {
            out_ptr[out_idx] = element * scale;
          }
        }
      }
    }
  }
}



template <>
__global__
void transform_kernel<__half>(int N, int C,
    int H, int W,  // original size
    int Hc, int Wc,  // cropped size
    bool param_mirror,
    int datum_height, int datum_width,  // offsets
    int crop_size, Phase phase,
    size_t sizeof_element,
    const __half* in,
    __half* out,  // buffers
    float scale,
    int has_mean_file,
    int has_mean_values,
    float* mean,
    const unsigned int *random_numbers) {
  const int c = blockIdx.y;

  // loop over images
  for (int n = blockIdx.x; n < N; n += gridDim.x) {
    // get mirror and offsets
    unsigned int rand1 = random_numbers[n*3    ];
    unsigned int rand2 = random_numbers[n*3 + 1];
    unsigned int rand3 = random_numbers[n*3 + 2];

    bool mirror = param_mirror && (rand1 % 2);
    int h_off = 0, w_off = 0;
    if (crop_size) {
      if (phase == TRAIN) {
        h_off = rand2 % (datum_height - crop_size + 1);
        w_off = rand3 % (datum_width - crop_size + 1);
      } else {
        h_off = (datum_height - crop_size) / 2;
        w_off = (datum_width - crop_size) / 2;
      }
    }

    const uint8_t *in_ptri;
    const float *in_ptrf;
    // offsets into start of (image, channel) = (n, c)
    // channel is handled by blockIdx.y
    // Initial offset per Dtype:
    const __half *in_ptr  = &in[n*C*H*W];
    // Element-specific offset to a channel c
    if (sizeof_element == sizeof(uint8_t)) {
      in_ptri = reinterpret_cast<const uint8_t*>(in_ptr);
      in_ptri += c*H*W;
    } else if (sizeof_element == sizeof(float)) {
      in_ptrf = reinterpret_cast<const float*>(in_ptr);
      in_ptrf += c*H*W;
    } else {
      in_ptr += c*H*W;
    }

    __half* out_ptr = &out[n*C*Hc*Wc + c*Hc*Wc];
    float element;
    // loop over pixels using threads
    for (int h = threadIdx.y; h < Hc; h += blockDim.y) {
      for (int w = threadIdx.x; w < Wc; w += blockDim.x) {
        // get the indices for in, out buffers
        int in_idx  = (h_off + h) * W + w_off + w;
        int out_idx = mirror ? h * Wc + (Wc - 1 - w) : h * Wc + w;

        if (sizeof_element == sizeof(uint8_t)) {
          element = in_ptri[in_idx];
        } else if (sizeof_element == sizeof(float)) {
          element = in_ptrf[in_idx];
        } else {
          element = __half2float(in_ptr[in_idx]);
        }

        // perform the transform
        if (has_mean_file) {
          out_ptr[out_idx] = float2half_clip((element - mean[c*H*W + in_idx]) * scale);
        } else {
          if (has_mean_values) {
            out_ptr[out_idx] = float2half_clip((element - mean[c]) * scale);
          } else {
            out_ptr[out_idx] = float2half_clip(element * scale);
          }
        }
      }
    }
  }
}


template <typename Dtype>
void DataTransformer<Dtype>::TransformGPU(int N, int C, int H, int W,
    size_t sizeof_element,
    const Dtype *in, Dtype *out,
    const unsigned int *random_numbers) {
  const int datum_channels = C;
  const int datum_height = H;
  const int datum_width = W;

  const int crop_size = param_.crop_size();
  float scale = param_.scale();
  const bool mirror = param_.mirror();
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  float* mean = nullptr;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    // no need to check equality anymore
    // datum_{height, width} are _output_ not input
    mean = data_mean_.mutable_gpu_data();
  }

  if (has_mean_values) {
    if (mean_values_gpu_.empty()) {
      CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
          << "Specify either 1 mean_value or as many as channels: "
          << datum_channels;
      if (datum_channels > 1 && mean_values_.size() == 1) {
        // Replicate the mean_value for simplicity
        for (int c = 1; c < datum_channels; ++c) {
          mean_values_.push_back(mean_values_[0]);
        }
      }
      mean_values_gpu_.reserve(sizeof(float) * mean_values_.size());
      caffe_copy(static_cast<int>(mean_values_.size()), &mean_values_.front(),
          reinterpret_cast<float*>(mean_values_gpu_.data()));
    }
    mean = reinterpret_cast<float*>(mean_values_gpu_.data());
  }

  int height = datum_height;
  int width = datum_width;

  if (crop_size) {
    height = crop_size;
    width = crop_size;
  }

  dim3 grid(N, C);
  dim3 block(16, 16);
  cudaStream_t stream = Caffe::th_stream_aux(Caffe::STREAM_ID_TRANSFORMER);

  transform_kernel<Dtype>
    <<< grid, block, 0, stream >>>(N, C, H, W,
        height, width,
        param_.mirror(),
        datum_height, datum_width,
        crop_size, phase_,
        sizeof_element,
        in, out,
        scale,
        static_cast<int>(has_mean_file),
        static_cast<int>(has_mean_values),
        mean, random_numbers);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template void DataTransformer<float>::TransformGPU(int, int, int, int,
    size_t, const float*, float*, const unsigned int*);
template void DataTransformer<double>::TransformGPU(int, int, int, int,
    size_t, const double*, double*, const unsigned int*);
template void DataTransformer<float16>::TransformGPU(int, int, int, int,
    size_t, const float16*, float16*, const unsigned int*);

}  // namespace caffe
