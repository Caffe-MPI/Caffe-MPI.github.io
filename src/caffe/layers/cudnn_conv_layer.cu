#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void CuDNNConvolutionLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* weight = this->blobs_[0]->template gpu_data<Ftype>();
  for (int i = 0; i < bottom.size(); ++i) {
    const Ftype* bottom_data = bottom[i]->gpu_data<Ftype>();
    Ftype* top_data = top[i]->mutable_gpu_data<Ftype>();
    // Forward through cuDNN in parallel over groups.
    const size_t gsize = workspace_.size() / groups();
    CHECK(is_even(gsize));
    for (int g = 0; g < this->group_; ++g) {
      unsigned char* pspace = static_cast<unsigned char*>(workspace_.data()) + gsize * idxg(g);
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(idxg(g)),
          cudnn::dataType<Ftype>::one, fwd_bottom_descs_[i], bottom_data + bottom_offset_ * g,
          fwd_filter_desc_, weight + this->weight_offset_ * g,
          fwd_conv_descs_[i], fwd_algo_[i], pspace, gsize,
          cudnn::dataType<Ftype>::zero, fwd_top_descs_[i], top_data + top_offset_ * g));
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    for (int ig = 0; ig < groups(); ++ig) {
      CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(ig)));
    }

    if (this->bias_term_) {
      const Ftype* bias_data = this->blobs_[1]->template gpu_data<Ftype>();
      for (int g = 0; g < this->group_; ++g) {
        CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(idxg(g)),
            cudnn::dataType<Ftype>::one,
            fwd_bias_desc_, bias_data + bias_offset_ * g,
            cudnn::dataType<Ftype>::one,
            fwd_top_descs_[i], top_data + top_offset_ * g));
      }
      // Synchronize the work across groups, each of which went into its own stream
      // NOLINT_NEXT_LINE(whitespace/operators)
      for (int g = 0; g < groups(); ++g) {
        CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(g)));
      }
    }
  }  // end of for i

  const Solver* psolver = this->parent_solver();
  if (psolver == nullptr || psolver->is_iter_size_complete()) {
    // Possibly use faster algorithms by allowing larger workspace.
    use_modest_workspace_ = false;
  }
}

template <typename Ftype, typename Btype>
void CuDNNConvolutionLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const size_t gsize = workspace_.size() / groups();
  CHECK(is_even(gsize));

  // compute dE/dB = sum_c(dE/dy)
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    Btype* bias_diff = this->blobs_[1]->template mutable_gpu_diff<Btype>();
    for (int i = 0; i < top.size(); ++i) {
      Btype* top_diff = top[i]->mutable_gpu_diff<Btype>();
      // in parallel over groups
      for (int g = 0; g < this->group_; ++g) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(idxg(g)),
            cudnn::dataType<Btype>::one, bwd_top_descs_[i], top_diff + top_offset_ * g,
            cudnn::dataType<Btype>::one, bwd_bias_desc_, bias_diff + bias_offset_ * g));
      }  // end of groups
      // Synchronize the work across groups, each of which went into its own stream
      // NOLINT_NEXT_LINE(whitespace/operators)
      for (int g = 0; g < groups(); ++g) {
        CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(g)));
      }
    }  // end of i
  }  // end of dB

  // compute dE/dW = dY * X
  if (this->param_propagate_down_[0]) {
    Btype* weight_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
    for (int i = 0; i < top.size(); ++i) {
      Btype* top_diff = top[i]->mutable_gpu_diff<Btype>();
      const Btype* bottom_data = bottom[i]->gpu_data<Btype>();
      // Backward through cuDNN in parallel over groups and gradients.
      for (int g = 0; g < this->group_; ++g) {
        unsigned char* pspace = static_cast<unsigned char*>(workspace_.data()) + gsize * idxg(g);
        // Gradient w.r.t. weights.
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(Caffe::cudnn_handle(idxg(g)),
            cudnn::dataType<Btype>::one, bwd_bottom_descs_[i], bottom_data + bottom_offset_ * g,
            bwd_top_descs_[i], top_diff + top_offset_ * g,
            bwd_conv_filter_descs_[i], bwd_filter_algo_[i], pspace, gsize,
            cudnn::dataType<Btype>::one, bwd_filter_desc_, weight_diff + this->weight_offset_ * g));
      }  // end of groups
      // Synchronize the work across groups, each of which went into its own stream
      // NOLINT_NEXT_LINE(whitespace/operators)
      for (int g = 0; g < groups(); ++g) {
        CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(g)));
      }
    }  // end of i
  }

  // Backward propagate grad wrt bottom data dE/dX= dE/dY * W
  const Btype* weight = this->blobs_[0]->template gpu_data<Btype>();
  for (int i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      // Backward in parallel over groups
      for (int g = 0; g < this->group_; ++g) {
        Btype* top_diff = top[i]->mutable_gpu_diff<Btype>();
        Btype* bottom_diff = bottom[i]->mutable_gpu_diff<Btype>();
        unsigned char* pspace = static_cast<unsigned char*>(workspace_.data()) + gsize * idxg(g);
        CUDNN_CHECK(cudnnConvolutionBackwardData(Caffe::cudnn_handle(idxg(g)),
            cudnn::dataType<Btype>::one, bwd_filter_desc_, weight + this->weight_offset_ * g,
            bwd_top_descs_[i], top_diff + top_offset_ * g,
            bwd_conv_data_descs_[i],
            bwd_data_algo_[i], pspace, gsize,
            cudnn::dataType<Btype>::zero, bwd_bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
      // Synchronize the work across groups.
      // NOLINT_NEXT_LINE(whitespace/operators)
      for (int g = 0; g < groups(); ++g) {
        CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(g)));
      }
    }  // end if propagate down
  }  // end for i
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
