#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Slice(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_slice_size = slice_size * top_slice_axis;
    const int slice_num = index / total_slice_size;
    const int slice_index = index % total_slice_size;
    const int bottom_index = slice_index +
        (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    if (forward) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}

template <typename Ftype, typename Btype>
void SliceLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
        const vector<Blob*>& top) {
  if (top.size() == 1) { return; }
  int offset_slice_axis = 0;
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = true;
  for (int i = 0; i < top.size(); ++i) {
    Ftype* top_data = top[i]->mutable_gpu_data<Ftype>();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
    Slice  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        nthreads, bottom_data, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, top_data);
    offset_slice_axis += top_slice_axis;
  }
}

template <typename Ftype, typename Btype>
void SliceLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0] || top.size() == 1) { return; }
  int offset_slice_axis = 0;
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = false;
  for (int i = 0; i < top.size(); ++i) {
    const Btype* top_diff = top[i]->gpu_diff<Btype>();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
    Slice  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        nthreads, top_diff, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, bottom_diff);
    offset_slice_axis += top_slice_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(SliceLayer);

}  // namespace caffe
