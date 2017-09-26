#ifndef CAFFE_UTIL_GPU_UTIL_H_
#define CAFFE_UTIL_GPU_UTIL_H_

#include "caffe/util/float16.hpp"
#include "caffe/util/gpu_math_functions.cuh"

namespace caffe {

template <typename Dtype>
inline __device__ Dtype caffe_gpu_atomic_add(const Dtype val, Dtype* address);

template <>
inline __device__
float caffe_gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

// double atomicAdd implementation taken from:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
template <>
inline __device__
double caffe_gpu_atomic_add(const double val, double* address) {
  unsigned long long int* address_as_ull =  // NOLINT(runtime/int)
      // NOLINT_NEXT_LINE(runtime/int)
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;  // NOLINT(runtime/int)
  unsigned long long int assumed;  // NOLINT(runtime/int)
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

template <>
inline __device__
float16 caffe_gpu_atomic_add(const float16 val, float16* address) {
// TODO check for FP16 implementation in future CUDA releases
// See atomicA in cudnn_ops.hxx

  // The size of 'unsigned' should be 4B, twice the size of 'half'.
  union U {
    unsigned u;
    // NOLINT_NEXT_LINE(runtime/arrays)
    ::half h[sizeof(unsigned) == 2 * sizeof(::half) ? 2 : -1];
    __device__ U() {}
  };
  unsigned *aligned_address = (unsigned int *)(((uintptr_t) address) & ~(uintptr_t)0x2);
  unsigned idx = (address == (float16*) aligned_address ? 0 : 1);
  U old_val, new_val, assumed;
  old_val.u = *aligned_address;
  float16 ret = *address;

  do {
    assumed.u = old_val.u;
    new_val.u = old_val.u;
    float tmp = __half2float(new_val.h[idx]);
    tmp += static_cast<float>(val);
    new_val.h[idx] = float2half_clip(tmp);
    old_val.u = atomicCAS(aligned_address, assumed.u, new_val.u);
  } while (assumed.u != old_val.u);
  return ret;
}

}  // namespace caffe

#endif  // CAFFE_UTIL_GPU_UTIL_H_
