#include <string>

#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/math_functions.hpp"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"
namespace caffe {

template<typename Gtype, typename Wtype>
__global__ void AdamRegUpdateAllAndClear(int N,
  Gtype* g, Wtype *w, Wtype* m, Wtype* v,
    float beta1, float beta2, float eps_hat, float local_rate,  float local_decay,
    bool reg_L2,  bool clear_grads) {
  CUDA_KERNEL_LOOP(i, N) {
    Wtype reg = reg_L2 ? w[i] : Wtype((Wtype(0) < w[i]) - (w[i] < Wtype(0)));
    Wtype gr = Wtype(g[i]) + reg * local_decay;
    Wtype mi = m[i] = m[i] * beta1 + gr * (Wtype(1.) - beta1);
    Wtype vi = v[i] = v[i] * beta2 + gr * gr * (Wtype(1.) - beta2);
    gr = local_rate * mi / (sqrt(vi) + eps_hat);
    w[i] -= gr;
    g[i] = clear_grads ? Gtype(0) : Gtype(gr);
  }
}
#pragma clang diagnostic pop

template<>
__global__ void AdamRegUpdateAllAndClear<half, half>(int N,
  half* g, half *w, half* m, half* v,
    float beta1, float beta2, float eps_hat, float local_rate, float local_decay,
    bool reg_L2,  bool clear_grads) {
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = __half2float(w[i]);
    float gf = __half2float(g[i]);
    float mf = __half2float(m[i]);
    float vf = __half2float(v[i]);

    float reg = reg_L2 ? wf : float((0.F < wf)-(wf < 0.F));
    gf += reg * local_decay;
    mf = beta1 * mf + (1.F - beta1)*gf;
    vf = beta2 * vf + (1.F - beta2)*gf*gf;
    gf = local_rate * mf / sqrt(vf + eps_hat);
    wf -= gf;

    w[i] = float2half_clip(wf);
    m[i] = float2half_clip(mf);
    v[i] = float2half_clip(vf);
    g[i] = clear_grads ? hz : float2half_clip(gf);
  }
}

template<typename Gtype, typename Wtype>
void adam_reg_update_and_clear_gpu(int N,
  Gtype* g,  Wtype *w, Wtype* m, Wtype* v,
  float beta1,  float beta2, float eps_hat, float local_rate,
    const std::string& reg_type, float local_decay, void *handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle() : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  AdamRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
      g, w, m, v,
      beta1, beta2, eps_hat, local_rate, local_decay, reg_type == "L2",  clear_grads);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void adam_reg_update_and_clear_gpu<float16, float16>(int N,
    float16 *g, float16 *w, float16 *m, float16 *v,
  float beta1,  float beta2, float eps_hat, float local_rate,
  const std::string& reg_type, float local_decay, void *handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
        handle == nullptr ? Caffe::cublas_handle() : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  AdamRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
      reinterpret_cast<half*>(g), reinterpret_cast<half*>(w),
      reinterpret_cast<half*>(m), reinterpret_cast<half*>(v),
      beta1, beta2, eps_hat, local_rate, local_decay, reg_type == "L2",  clear_grads);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}


template void adam_reg_update_and_clear_gpu<float16, float>(int, float16*, float*, float*,
    float*, float, float, float, float, const std::string&, float, void*, bool);
template void adam_reg_update_and_clear_gpu<float16, double>(int, float16*, double*, double*,
    double*, float, float, float, float, const std::string&, float, void*, bool);
template void adam_reg_update_and_clear_gpu<float, float>(int, float*, float*, float*,
    float*, float, float, float, float, const std::string&, float,  void*, bool);
template void adam_reg_update_and_clear_gpu<float, double>(int, float*, double*, double*,
    double*, float, float, float, float, const std::string&, float, void*, bool);
template void adam_reg_update_and_clear_gpu<float, float16>(int, float*, float16*, float16*,
    float16*, float, float, float, float, const std::string&, float, void*, bool);
template void adam_reg_update_and_clear_gpu<double, float>(int, double*, float*, float*,
    float*, float, float, float, float, const std::string&, float, void*, bool);
template void adam_reg_update_and_clear_gpu<double, double>(int, double*, double*, double*,
    double*, float, float, float, float, const std::string&, float, void*, bool);
template void adam_reg_update_and_clear_gpu<double, float16>(int, double*, float16*, float16*,
    float16*, float, float, float, float, const std::string&, float, void*, bool);
}  // namespace caffe
