#include <vector>
#include <device_launch_parameters.h>

#include "caffe/filler.hpp"
#include "caffe/layers/embed_layer.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void EmbedForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* weight, const int M, const int N, const int K,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(top_index, nthreads) {
    const int n = top_index / N;
    const int d = top_index % N;
    const int index = static_cast<int>(static_cast<double>(bottom_data[n]));
    const int weight_index = index * N + d;
    top_data[top_index] = weight[weight_index];
  }
}

template <typename Dtype>
__global__ void EmbedBackward(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_diff, const int M, const int N, const int K,
    Dtype* weight_diff);

template <typename Dtype>
__global__ void EmbedBackward(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_diff, const int M, const int N, const int K,
    Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(top_index, nthreads) {
    const int n = top_index / N;
    const int d = top_index % N;
    const int index = static_cast<int>(static_cast<double>(bottom_data[n]));
    const int weight_index = index * N + d;
    caffe_gpu_atomic_add(top_diff[top_index], weight_diff + weight_index);
  }
}

template <typename Ftype, typename Btype>
void EmbedLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const Ftype* weight = this->blobs_[0]->template gpu_data<Ftype>();
  const int count = top[0]->count();
  EmbedForward  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
      count, bottom_data, weight, M_, N_, K_, top_data);
  if (bias_term_) {
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, Ftype(1),
        bias_multiplier_.template gpu_data<Ftype>(),
        this->blobs_[1]->template gpu_data<Ftype>(), Ftype(1), top_data);
  }
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

template <typename Ftype, typename Btype>
void EmbedLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  CHECK(!propagate_down[0]) << "Can't backpropagate to EmbedLayer input.";
  if (this->param_propagate_down_[0]) {
    const int top_count = top[0]->count();
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
    Btype* weight_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
    EmbedBackward  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        top_count, bottom_data, top_diff, M_, N_, K_, weight_diff);
    CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    Btype* bias_diff = this->blobs_[1]->template mutable_gpu_diff<Btype>();
    caffe_gpu_gemv(CblasTrans, M_, N_, Btype(1), top_diff,
        bias_multiplier_.template gpu_data<Btype>(), Btype(1), bias_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(EmbedLayer);

}  // namespace caffe
