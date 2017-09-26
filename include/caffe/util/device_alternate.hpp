#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#ifdef CPU_ONLY  // CPU-only Caffe.

#include <vector>

// Stub out GPU calls as unavailable.

#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#define STUB_GPU(classname) \
template <typename Ftype, typename Btype> \
void classname<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom, \
    const vector<Blob*>& top) { NO_GPU; } \
template <typename Ftype, typename Btype> \
void classname<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob*>& bottom) { NO_GPU; }

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Ftype, typename Btype> \
void classname<Ftype, Btype>::funcname##_##gpu(const vector<Blob*>& bottom, \
    const vector<Blob*>& top) { NO_GPU; }

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Ftype, typename Btype> \
void classname<Ftype, Btype>::funcname##_##gpu(const vector<Blob*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob*>& bottom) { NO_GPU; }

#define STUB_GPU_FORWARD1(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob*>& bottom, \
    const vector<Blob*>& top) { NO_GPU; }

#else  // Normal GPU + CPU Caffe.

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifndef NO_NVML
  #include <nvml.h>
#endif
#include <sched.h>

//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace caffe {

// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;
const int CAFFE_CUDA_NUM_THREADS_HALF = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
inline int CAFFE_GET_BLOCKS_HALF(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS_HALF - 1) /
      CAFFE_CUDA_NUM_THREADS_HALF;
}


#ifndef NO_NVML
namespace nvml {

// We might move this to Caffe TLS but we have to make sure that
// this one gets initialized immediately after thread start.
// Also, it's better to run this on current device (note that Caffe ctr
// might be executed somewhere else). So, let's keep it risk free.
struct NVMLInit {
  NVMLInit() {
    if (nvmlInit() != NVML_SUCCESS) {
      LOG(ERROR) << "NVML failed to initialize";
    } else {
      LOG(INFO) << "NVML initialized on thread " << std::this_thread::get_id();
    }
  }
  ~NVMLInit() {
    nvmlShutdown();
  }

  nvmlDevice_t device_;
  static std::mutex m_;
};

void setCpuAffinity(unsigned int rank);

}  // namespace nvml
#endif  // NO_NVML

}  // namespace caffe
#endif  // CPU_ONLY
#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
