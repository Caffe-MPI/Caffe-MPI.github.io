#include <device_launch_parameters.h>

#include "caffe/common.hpp"
#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/math_functions.hpp"

namespace caffe {

///////////////////////////////////// ASUM REDUCTION ///////////////////////////////////

template<unsigned int BlockSize, typename TR>
__device__ void asum_reduce_block(volatile TR *sdata, TR my_sum, unsigned int tid) {
  volatile TR* st = sdata + tid;
  tassign(st, my_sum);
  __syncthreads();

  // do reduction in shared mem
  if (BlockSize >= 512) {
    if (tid < 256) {
      tsum_replace(st, sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (BlockSize >= 256) {
    if (tid < 128) {
      tsum_replace(st, sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (BlockSize >= 128) {
    if (tid < 64) {
      tsum_replace(st, sdata[tid + 64]);
    }
    __syncthreads();
  }
  if (tid < 32) {
    for (int i = 32; i > 0; i >>= 1) {
      tsum_replace(st, sdata[tid + i]);
    }
  }
}


// Global variable used by amax_reduce_kernel to count how many blocks have finished
__device__ unsigned int asum_blocks_count_f = 0;
__device__ unsigned int asum_blocks_count_d = 0;
__device__ unsigned int asum_blocks_count_h = 0;

template<typename T>
__device__ __inline__
unsigned int* asum_blocks_count_ptr();
template<>
__device__ __inline__
unsigned int* asum_blocks_count_ptr<float>() {
  return &asum_blocks_count_f;
}
template<>
__device__ __inline__
unsigned int* asum_blocks_count_ptr<double>() {
  return &asum_blocks_count_d;
}
template<>
__device__ __inline__
unsigned int* asum_blocks_count_ptr<half2>() {
  return &asum_blocks_count_h;
}

template<typename T>
cudaError_t set_asum_blocks_count(unsigned int cnt);
template<>
cudaError_t set_asum_blocks_count<float>(unsigned int cnt) {
  return cudaMemcpyToSymbolAsync(asum_blocks_count_f, &cnt, sizeof(unsigned int), 0,
      cudaMemcpyHostToDevice, Caffe::thread_stream());
}
template<>
cudaError_t set_asum_blocks_count<double>(unsigned int cnt) {
  return cudaMemcpyToSymbolAsync(asum_blocks_count_d, &cnt, sizeof(unsigned int), 0,
      cudaMemcpyHostToDevice, Caffe::thread_stream());
}
template<>
cudaError_t set_asum_blocks_count<half2>(unsigned int cnt) {
  return cudaMemcpyToSymbolAsync(asum_blocks_count_h, &cnt, sizeof(unsigned int), 0,
      cudaMemcpyHostToDevice, Caffe::thread_stream());
}

template<typename T>
__device__ __inline__
void reset_asum_blocks_count();
template<>
void reset_asum_blocks_count<float>() {
  asum_blocks_count_f = 0;
}
template<>
__device__ __inline__
void reset_asum_blocks_count<double>() {
  asum_blocks_count_d = 0;
}
template<>
__device__ __inline__
void reset_asum_blocks_count<half2>() {
  asum_blocks_count_h = 0;
}

template<unsigned int BlockSize, bool IsPow2, typename T, typename TR>
__device__ void asum_reduce_blocks(const T *in, TR *out, unsigned int n) {
  struct __dyn_shmem__<n_bytes<sizeof(TR)>> asum_blocks_shmem;
  TR* partial_asum = reinterpret_cast<TR*>(asum_blocks_shmem.getPtr());

  // first level of reduction:
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * BlockSize * 2 + threadIdx.x;
  unsigned int gridSize = BlockSize * 2 * gridDim.x;
  T t1, t2;
  TR my_sum = tzero<TR>();
  // We reduce multiple elements per thread. The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  while (i < n) {
    t1 = tabs(in[i]);
    if (IsPow2 || i + BlockSize < n) {
      t2 = tabs(in[i + BlockSize]);
      tsum_replace(&my_sum, tsum<T, TR>(t1, t2));
    } else {
      tsum_replace(&my_sum, t1);
    }
    i += gridSize;
  }

  // do reduction in shared mem
  asum_reduce_block<BlockSize>(partial_asum, my_sum, tid);
  // write result for this block to global mem
  if (tid == 0) {
    out[blockIdx.x] = partial_asum[0];
  }
}

template<unsigned int BlockSize, bool IsPow2, typename T, typename TR>
__global__ void asum_reduce_kernel(unsigned int n, const T *in, TR *out) {
  asum_reduce_blocks<BlockSize, IsPow2>(in, out, n);

  if (gridDim.x > 1) {
    const unsigned int tid = threadIdx.x;
    struct __dyn_shmem__<n_bytes<sizeof(TR)>> asum_reduce_shmem;
    TR* partial_asum = reinterpret_cast<TR*>(asum_reduce_shmem.getPtr());
    __shared__ bool last_asum_reduce_block;

    // wait until all outstanding memory instructions in this thread are finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0) {
      unsigned int ticket = atomicInc(asum_blocks_count_ptr<T>(), gridDim.x);
      last_asum_reduce_block = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    // The last block sums the results of all other blocks
    if (last_asum_reduce_block) {
      int i = tid;
      TR my_sum = tzero<TR>();

      while (i < gridDim.x) {
        tsum_replace(&my_sum, out[i]);
        i += BlockSize;
      }
      asum_reduce_block<BlockSize>(partial_asum, my_sum, tid);
      if (tid == 0) {
        out[0] = partial_asum[0];
        // reset blocks count so that next run succeeds
        reset_asum_blocks_count<T>();
      }
    }
  }
}

template<typename T, typename TR>
void gpu_asum_t(const int n, const T* x, TR* sum) {
  cudaStream_t stream = Caffe::thread_stream();
  const bool po2 = is_pow2(n);
  // See kernel for details
  CHECK_LE(CAFFE_CUDA_NUM_THREADS_HALF, 512);
  CHECK_GE(CAFFE_CUDA_NUM_THREADS_HALF, 128);
  const int threadsPerCta = CAFFE_CUDA_NUM_THREADS_HALF;
  const int nbrCtas = CAFFE_GET_BLOCKS_HALF(n);
  const int reduction_size_sum = (nbrCtas + 1) * sizeof(TR);
  TR* dev_ptr_sum = reinterpret_cast<TR*>(GPUMemory::pinned_buffer(reduction_size_sum));
  if (po2 && n > CAFFE_CUDA_NUM_THREADS_HALF) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    asum_reduce_kernel<CAFFE_CUDA_NUM_THREADS_HALF, true><<<nbrCtas, threadsPerCta,
        threadsPerCta * sizeof(TR) + sizeof(bool), stream>>>
            ((unsigned int)n, x, dev_ptr_sum);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    asum_reduce_kernel<CAFFE_CUDA_NUM_THREADS_HALF, false><<<nbrCtas, threadsPerCta,
        threadsPerCta * sizeof(TR) + sizeof(bool), stream>>>
            ((unsigned int)n, x, dev_ptr_sum);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  *sum = dev_ptr_sum[0];
}

template<>
void caffe_gpu_asum<float16, float>(const int n, const float16* x, float* sum) {
  // For odd counts we allocate extra element to speed up kernels.
  // We have to keep it clean.
  cudaStream_t stream = Caffe::thread_stream();
  if (n & 1) {
    clean_last_element(const_cast<float16*>(x) + n, stream);
  }
  const int n2 = even(n) / 2;
  static cudaError_t status = set_asum_blocks_count<half2>(0U);  // needed just 1 time
  CUDA_CHECK(status);
  gpu_asum_t(n2, reinterpret_cast<const half2*>(x), sum);
}
template<>
void caffe_gpu_asum<float16, double>(const int n, const float16* x, double* sum) {
  float sf;
  caffe_gpu_asum(n, x, &sf);
  *sum = sf;
}
template<>
void caffe_gpu_asum<float16, float16>(const int n, const float16* x, float16* sum) {
  float sf;
  caffe_gpu_asum(n, x, &sf);
  *sum = sf;
}


}  // namespace caffe
