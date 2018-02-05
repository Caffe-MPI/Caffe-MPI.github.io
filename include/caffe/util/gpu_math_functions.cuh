#ifndef INCLUDE_CAFFE_GPU_MATH_FUNCTIONS_H_
#define INCLUDE_CAFFE_GPU_MATH_FUNCTIONS_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <math_functions.h>
#include <driver_types.h>

#include "caffe/util/half.cuh"

__device__ __inline__ half inf_clip(half h) {
  const int isi = __hisinf(h);
  if (isi > 0) {
    // Exponent all ones except LSB (0x1e), mantissa is all ones (0x3ff)
    h.setx(0x7bffU);
  } else if (isi < 0) {
    // As above, negated
    h.setx(0x7bffU ^ 0x8000U);
  }
  return h;
}

__device__ __inline__ half float2half_clip(float a) {
#ifdef OLD_CUDA_HALF_IMPL
  half h;
  h.setx(__float2half_rn(a));
  return inf_clip(h);
#else
  return inf_clip(__float2half_rn(a));
#endif
}

__device__ __inline__
half2 float22half2_clip(float2 a) {
  half2 h = __float22half2_rn(a);
  return __halves2half2(inf_clip(__low2half(h)), inf_clip(__high2half(h)));
}

__device__ __inline__
bool hlt(half a, half b) {
#if __CUDA_ARCH__ >= 530
return __hlt(a, b);
#else
return __half2float(a) < __half2float(b);
#endif
}

__device__ __inline__
half2 hmul2(half2 a, half2 b) {
#if __CUDA_ARCH__ >= 530
  return __hmul2(a, b);
#else
  float2 af = __half22float2(a);
  float2 bf = __half22float2(b);

  af.x *= bf.x;
  af.y *= bf.y;

  return float22half2_clip(af);
#endif
}

__device__ __inline__
half2 hge2(half2 a, half2 b) {
#if __CUDA_ARCH__ >= 530
  return __hge2(a, b);
#else
  float2 af = __half22float2(a);
  float2 bf = __half22float2(b);

  af.x = (af.x >= bf.x) ? 1.f : 0.f;
  af.y = (af.y >= bf.y) ? 1.f : 0.f;

  return float22half2_clip(af);
#endif
}

__device__ __inline__
half2 hadd2(half2 a, half2 b) {
#if __CUDA_ARCH__ >= 530
  return __hadd2(a, b);
#else
  float2 af = __half22float2(a);
  float2 bf = __half22float2(b);

  af.x += bf.x;
  af.y += bf.y;

  return float22half2_clip(af);
#endif
}

__device__ __inline__
half h_abs(half a) {
  a.setx(a.x() & 0x7FFF);
  return a;
}

__device__ __inline__
half2 h2_abs(half2 a) {
#ifdef OLD_CUDA_HALF_IMPL
  a.x &= 0x7FFF7FFFU;
#else
  a.set_lo(h_abs(a.lo()));
  a.set_hi(h_abs(a.hi()));
#endif
  return a;
}

// a <- max(a,b)
__device__ __inline__
void h2_max_replace(volatile half2 *a, half2 b) {
  half2 cmp = hge2(*const_cast<half2*>(a), b);  // 00 01 10 11
  bool al = cmp.lo();                // true: a.low >= b.low
  bool ah = cmp.hi();               // true: a.high >= b.high
  if (al) {
    if (ah) {
      // (a.low,a.high)
    } else {
      // (a.low,b.high)
      const_cast<half2*>(a)->set_hi(b.hi());
    }
  } else {
    if (ah) {
      // (b.low,a.high)
      const_cast<half2*>(a)->set_lo(b.lo());
    } else {
      // (b.low,b.high)
      *const_cast<half2*>(a) = b;
    }
  }
}

// <- max(a,b)
__device__ __inline__
half2 h2_max(half2 a, half2 b) {
  half2 m = a;
  h2_max_replace(&m, b);
  return m;
}


template<typename T> struct __dyn_shmem__ {
  __device__
  T* getPtr() {
    return NULL;
  }
};
template<> struct __dyn_shmem__<float> {
  __device__
  float* getPtr() {
    extern __shared__ float Sptr[];
    return Sptr;
  }
};
template<> struct __dyn_shmem__<double> {
  __device__
  double* getPtr() {
    extern __shared__ double Dptr[];
    return Dptr;
  }
};
template<> struct __dyn_shmem__<half2> {
  __device__
  half2* getPtr() {
    extern __shared__ half2 Hptr[];
    return Hptr;
  }
};
template<> struct __dyn_shmem__<int> {
  __device__
  int* getPtr() {
    extern __shared__ int Iptr[];
    return Iptr;
  }
};
template<> struct __dyn_shmem__<unsigned int> {
  __device__
  unsigned int* getPtr() {
    extern __shared__ unsigned int UIptr[];
    return UIptr;
  }
};

template <int N>
struct n_bytes {
  char content[N];
};

#define CAFFE_GPU_SHMEM(N)                    \
template<> struct __dyn_shmem__<n_bytes<N>> { \
  __device__ n_bytes<N>* getPtr() {           \
    extern __shared__ n_bytes<N> ptr##N[];    \
    return ptr##N;                            \
  }                                           \
}

CAFFE_GPU_SHMEM(2);
CAFFE_GPU_SHMEM(4);
CAFFE_GPU_SHMEM(6);
CAFFE_GPU_SHMEM(8);
CAFFE_GPU_SHMEM(10);
CAFFE_GPU_SHMEM(12);
CAFFE_GPU_SHMEM(14);
CAFFE_GPU_SHMEM(16);


template<typename T>
__device__ __inline__
T tabs(T t);
template<>
__device__ __inline__
int tabs<int>(int t) {
  return abs(t);
}
template<>
__device__ __inline__
float tabs<float>(float t) {
  return fabs(t);
}
template<>
__device__ __inline__
double tabs<double>(double t) {
  return fabs(t);
}
template<>
__device__ __inline__
half2 tabs<half2>(half2 t) {
  return h2_abs(t);
}

template<typename T>
__device__ __inline__
T tzero() {
  return 0;
}
template<>
__device__ __inline__
half2 tzero<half2>() {
  return half2(half(), half());
}

template<typename T>
__device__ __inline__
int non_zero_count(T a) {
  return a == 0 ? 0 : 1;
}
template<>
__device__ __inline__
int non_zero_count<float>(float a) {
  return a == 0.F ? 0 : 1;
}
template<>
__device__ __inline__
int non_zero_count<double>(double a) {
  return a == 0. ? 0 : 1;  // DBL_EPSILON?
}
template<>
__device__ __inline__
int non_zero_count<half2>(half2 a) {
  return (a.lo() ? 1 : 0) + (a.hi() ? 1 : 0);
}

template<typename T, typename TR>
__device__ __inline__
TR tmax(T a, T b) {
  return TR(a > b ? a : b);
}
template<>
__device__ __inline__
half2 tmax<half2, half2>(half2 a, half2 b) {
  return h2_max(a, b);
}
template<>
__device__ __inline__
float tmax<half2, float>(half2 a, half2 b) {
  float2 fm = __half22float2(h2_max(a, b));
  return tmax<float, float>(fm.x, fm.y);
}
template<>
__device__ __inline__
double tmax<half2, double>(half2 a, half2 b) {
  float2 fm = __half22float2(h2_max(a, b));
  return tmax<float, float>(fm.x, fm.y);
}

template<typename T, typename TR>
__device__ __inline__
TR tsum(T a, T b) {
  return TR(a + b);
}
template<>
__device__ __inline__
half2 tsum<half2, half2>(half2 a, half2 b) {
  return hadd2(a, b);
}
template<>
__device__ __inline__
float tsum<half2, float>(half2 a, half2 b) {
  half2 h = hadd2(a, b);
  float2 f = __half22float2(h);
  return f.x + f.y;
}
template<>
__device__ __inline__
double tsum<half2, double>(half2 a, half2 b) {
  return tsum<half2, float>(a, b);
}

template<typename T>
__device__ __inline__
void tassign(volatile T *a, T b) {
  *a = b;
}

template<typename T, typename TR>
__device__ __inline__
void tmax_replace(volatile TR *a, T b) {
  if (b > *a) {
    *a = b;
  }
}
template<>
__device__ __inline__
void tmax_replace<half2, half2>(volatile half2 *a, half2 b) {
  h2_max_replace(a, b);
}
template<>
__device__ __inline__
void tmax_replace<half2, float>(volatile float *a, half2 b) {
  float2 f = __half22float2(b);
  if (f.x > f.y) {
    tmax_replace(a, f.x);
  } else {
    tmax_replace(a, f.y);
  }
}
template<>
__device__ __inline__
void tmax_replace<half2, double>(volatile double *a, half2 b) {
  float2 f = __half22float2(b);
  if (f.x > f.y) {
    tmax_replace(a, f.x);
  } else {
    tmax_replace(a, f.y);
  }
}

template<typename T, typename TR>
__device__ __inline__
void tsum_replace(volatile TR *a, T b) {
  *a += b;
}
template<>
__device__ __inline__
void tsum_replace<half2, half2>(volatile half2 *a, half2 b) {
  *const_cast<half2*>(a) = hadd2(*const_cast<half2*>(a), b);
}
template<>
__device__ __inline__
void tsum_replace<half2, float>(volatile float *a, half2 b) {
  float2 f = __half22float2(b);
  *a += f.x + f.y;
}
template<>
__device__ __inline__
void tsum_replace<half2, double>(volatile double *a, half2 b) {
  float2 f = __half22float2(b);
  *a += f.x + f.y;
}

#endif  // INCLUDE_CAFFE_GPU_MATH_FUNCTIONS_H_
