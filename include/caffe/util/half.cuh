/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// Conversion from/to 16-bit floating point (half-precision).
#ifndef INCLUDE_CAFFE_UTIL_HALF_CUH_
#define INCLUDE_CAFFE_UTIL_HALF_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <driver_types.h>

#if !defined(OLD_CUDA_HALF_IMPL)
  #if CUDA_VERSION < 9000
    #define OLD_CUDA_HALF_IMPL
  #endif
#endif

/**
 * GPU-specific float16 data type
 */
class alignas(2) half : public __half {
 public:
  __host__ __device__
  half() {
#ifdef OLD_CUDA_HALF_IMPL
    __half::x = 0U;
#else
    __x = 0U;
#endif
  }

  __host__ __device__
  half(const half& other) {
#ifdef OLD_CUDA_HALF_IMPL
    __half::x = other.x();
#else
    __x = other.__x;
#endif
  }

  __host__ __device__
  half(half&& other) {
#ifdef OLD_CUDA_HALF_IMPL
    __half::x = other.x();
#else
    __x = other.__x;
#endif
  }

  __host__ __device__
  half(const __half& other)
      : __half(other) {}

  __host__ __device__
  half(__half&& other)
      : __half(std::move(other)) {}

  __host__ __device__
  half& operator = (const half& other) {
#ifdef OLD_CUDA_HALF_IMPL
    __half::x = other.x();
#else
    __x = other.__x;
#endif
    return *this;
  }

  __host__ __device__
  // NOLINT_NEXT_LINE(runtime/int)
  unsigned short x() const {
#ifdef OLD_CUDA_HALF_IMPL
    return __half::x;
#else
    return __x;
#endif
  }

  __host__ __device__
  // NOLINT_NEXT_LINE(runtime/int)
  half& setx(unsigned short x) {
#ifdef OLD_CUDA_HALF_IMPL
    __half::x = x;
#else
    __x = x;
#endif
    return *this;
  }

#ifdef OLD_CUDA_HALF_IMPL
  __host__ __device__
  operator bool() const {
    return (__half::x & 0x7fffU) != 0U;  // +0, -0
  }
#endif
};

struct alignas(4) half2 : public __half2 {
 public:
  __host__ __device__
  half2() {}

  __host__ __device__
  half2(const half2& other)
      : __half2(other) {}

  __host__ __device__
  half2(half2&& other)
      : __half2(std::move(other)) {}

  __host__ __device__
  half2(const __half2& other)
      : __half2(other) {}

  __host__ __device__
  half2(__half2&& other)
      : __half2(std::move(other)) {}

  __host__ __device__
  half2(const __half &l, const __half &h)
#ifdef OLD_CUDA_HALF_IMPL
      {
        __half2::x = l.x + (static_cast<unsigned int>(h.x) << 16);
      }
#else
      : __half2(l, h) {}
#endif

  __host__ __device__
  half lo() const {
#ifdef OLD_CUDA_HALF_IMPL
    half l;
    l.setx(__half2::x & 0xffffU);
    return l;
#else
    return x;
#endif
  }

  __host__ __device__
  half hi() const {
#ifdef OLD_CUDA_HALF_IMPL
    half h;
    // NOLINT_NEXT_LINE(runtime/int)
    h.setx(static_cast<unsigned short>(__half2::x >> 16));
    return h;
#else
    return y;
#endif
  }

  __host__ __device__
  half2& set_lo(half l) {
#ifdef OLD_CUDA_HALF_IMPL
    __half2::x = (__half2::x & 0xffff0000U) + l.x();
#else
    x = l;
#endif
    return *this;
  }

  __host__ __device__
  half2& set_hi(half h) {
#ifdef OLD_CUDA_HALF_IMPL
    __half2::x = (__half2::x & 0xffffU) + (h.x() << 16);
#else
    y = h;
#endif
    return *this;
  }

  __host__ __device__
  half2& operator = (const half2& other) {
#ifdef OLD_CUDA_HALF_IMPL
    __half2::x = other.x;
#else
    x = other.lo();
    y = other.hi();
#endif
    return *this;
  }
};

#if false
// TODO Clean later
__inline__ __device__ __host__ half habs(half h) {
  h.setx(h.x() & 0x7fffU);
  return h;
}

__inline__ __device__ __host__ half hneg(half h) {
  h.setx(h.x() ^ 0x8000U);
  return h;
}

__inline__ __device__ __host__ int ishnan(half h) {
  // When input is NaN, exponent is all ones and mantissa is non-zero.
  return (h.x() & 0x7c00U) == 0x7c00U && (h.x() & 0x03ffU) != 0;
}

__inline__ __device__ __host__ int ishinf(half h) {
  // When input is +/- inf, exponent is all ones and mantissa is zero.
  return (h.x() & 0x7c00U) == 0x7c00U && (h.x() & 0x03ffU) == 0;
}

__inline__ __device__ __host__ int ishequ(half x, half y) {
  return ishnan(x) == 0 && ishnan(y) == 0 &&
      (x.x() == y.x() || x.x() & 0x7fffU == y.x() & 0x7fffU);
}

// Returns 0.0000 in FP16 binary form
__inline__ __device__ __host__ half hzero() {
  half ret;
  ret.setx(0U);
  return ret;
}

// Returns 1.0000 in FP16 binary form
__inline__ __device__ __host__ half hone() {
  half ret;
  ret.setx(0x3c00U);
  return ret;
}

// Returns quiet NaN, the most significant fraction bit #9 is set
__inline__ __device__ __host__ half hnan() {
  half ret;
  ret.setx(0x7e00U);
  return ret;
}

// Largest positive FP16 value, corresponds to 6.5504e+04
__inline__ __device__ __host__ half hmax() {
  half ret;
  // Exponent all ones except LSB (0x1e), mantissa is all ones (0x3ff)
  ret.setx(0x7bffU);
  return ret;
}

// Smallest positive (normalized) FP16 value, corresponds to 6.1035e-05
__inline__ __device__ __host__ half hmin() {
  half ret;
  // Exponent is 0x01 (5 bits), mantissa is all zeros (10 bits)
  ret.setx(0x0400U);
  return ret;
}
#endif

#endif  // INCLUDE_CAFFE_UTIL_HALF_CUH_
