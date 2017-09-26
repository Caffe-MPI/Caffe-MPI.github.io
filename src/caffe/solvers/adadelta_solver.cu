#include <string>

#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"
template<typename Gtype, typename Wtype>
__global__ void AdaDeltaRegUpdateAllAndClear(int N,
    Gtype* g, Wtype *w, Wtype* h, Wtype* h2,
    float momentum, float delta, float local_rate, float local_decay, bool reg_L2,
    bool clear_grads) {
  CUDA_KERNEL_LOOP(i, N) {
    Wtype reg = reg_L2 ? w[i] : Wtype((Wtype(0) < w[i]) - (w[i] < Wtype(0)));
    Wtype gr = Wtype(g[i]) + reg * local_decay;
    Wtype hi = h[i] = momentum * h[i] + Wtype(1.F - momentum) * gr * gr;
    gr *= sqrt((h2[i] + delta) / (hi + delta));
    h2[i] = momentum * h2[i] + Wtype(1.F - momentum) * gr * gr;
    gr *= local_rate;
    w[i] -= gr;
    g[i] = clear_grads ? Gtype(0) : Gtype(gr);
  }
}
#pragma clang diagnostic pop

template<>
__global__ void AdaDeltaRegUpdateAllAndClear<half, half>(int N,
    half* g, half *w, half* h, half* h2,
    float momentum, float delta, float local_rate, float local_decay, bool reg_L2,
    bool clear_grads) {
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = __half2float(w[i]);
    float hf = __half2float(h[i]);
    float h2f = __half2float(h2[i]);
    float gr = __half2float(g[i]);

    float reg = reg_L2 ? wf : float((0.F < wf)-(wf < 0.F));
    gr +=  reg * local_decay;

    hf = momentum * hf + (1.F - momentum) * gr * gr;
    gr *= sqrt((h2f + delta) / (hf + delta));
    h2f = momentum * h2f + (1.F - momentum) * gr * gr;
    gr *= local_rate;
    wf -= gr;

    h[i]  = float2half_clip(hf);
    h2[i] = float2half_clip(h2f);
    w[i]  = float2half_clip(wf);
    g[i] = clear_grads ? hz : float2half_clip(gr);
  }
}

template<typename Gtype, typename Wtype>
void adadelta_reg_update_and_clear_gpu(int N,
  Gtype* g, Wtype* w, Wtype* h, Wtype* h2,
  float momentum,  float delta, float local_rate, const std::string& reg_type, float local_decay,
     void* handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle() : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  AdaDeltaRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N,
       g, w, h, h2,
       momentum, delta, local_rate, local_decay, reg_type == "L2", clear_grads);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void adadelta_reg_update_and_clear_gpu<float16, float16>(int N,
    float16 *g, float16 *w, float16 *h, float16 *h2,
  float momentum,  float delta, float local_rate, const std::string& reg_type, float local_decay,
    void* handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
        handle == nullptr ? Caffe::cublas_handle() : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  AdaDeltaRegUpdateAllAndClear<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>
      (N, reinterpret_cast<half*>(g), reinterpret_cast<half*>(w),
          reinterpret_cast<half*>(h), reinterpret_cast<half*>(h2),
          momentum, delta, local_rate, local_decay, reg_type == "L2", clear_grads);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template void adadelta_reg_update_and_clear_gpu<float16, float>(int, float16*, float*, float*,
    float*, float, float, float, const std::string&, float, void*, bool);
template void adadelta_reg_update_and_clear_gpu<float16, double>(int, float16*, double*, double*,
    double*, float, float, float, const std::string&, float, void*, bool);
template void adadelta_reg_update_and_clear_gpu<float, float>(int, float*, float*, float*,
    float*, float, float, float, const std::string&, float, void*, bool);
template void adadelta_reg_update_and_clear_gpu<float, double>(int, float*, double*, double*,
    double*, float, float, float, const std::string&, float, void*, bool);
template void adadelta_reg_update_and_clear_gpu<float, float16>(int, float*, float16*, float16*,
    float16*, float, float, float, const std::string&, float, void*, bool);
template void adadelta_reg_update_and_clear_gpu<double, float>(int, double*, float*, float*,
    float*, float, float, float, const std::string&, float, void*, bool);
template void adadelta_reg_update_and_clear_gpu<double, double>(int, double*, double*, double*,
    double*, float, float, float, const std::string&, float, void*, bool);
template void adadelta_reg_update_and_clear_gpu<double, float16>(int, double*, float16*, float16*,
    float16*, float, float, float, const std::string&, float, void*, bool);

}  // namespace caffe
