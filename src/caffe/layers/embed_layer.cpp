#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/embed_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void EmbedLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  N_ = this->layer_param_.embed_param().num_output();
  CHECK_GT(N_, 0) << "EmbedLayer num_output must be positive.";
  K_ = this->layer_param_.embed_param().input_dim();
  CHECK_GT(K_, 0) << "EmbedLayer input_dim must be positive.";
  bias_term_ = this->layer_param_.embed_param().bias_term();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights --
    // transposed from InnerProductLayer for spatial locality.
    vector<int> weight_shape(2);
    weight_shape[0] = K_;
    weight_shape[1] = N_;
    this->blobs_[0] = Blob::create<Ftype>(weight_shape);
    // fill the weights
    shared_ptr<Filler<Ftype> > weight_filler(GetFiller<Ftype>(
        this->layer_param_.embed_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1] = Blob::create<Ftype>(bias_shape);
      shared_ptr<Filler<Ftype> > bias_filler(GetFiller<Ftype>(
          this->layer_param_.embed_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Ftype, typename Btype>
void EmbedLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  // Figure out the dimensions
  M_ = bottom[0]->count();
  vector<int> top_shape = bottom[0]->shape();
  top_shape.push_back(N_);
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set<Ftype>(M_, Ftype(1), bias_multiplier_.template mutable_cpu_data<Ftype>());
  }
}

template <typename Ftype, typename Btype>
void EmbedLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  const Ftype* weight = this->blobs_[0]->template cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  int index;
  for (int n = 0; n < M_; ++n) {
    index = static_cast<int>(bottom_data[n]);
    DCHECK_GE(index, 0);
    DCHECK_LT(index, K_);
    DCHECK_EQ(static_cast<Ftype>(index), bottom_data[n]) << "non-integer input";
    caffe_copy(N_, weight + index * N_, top_data + n * N_);
  }
  if (bias_term_) {
    const Ftype* bias = this->blobs_[1]->template cpu_data<Ftype>();
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, Ftype(1),
        bias_multiplier_.template cpu_data<Ftype>(), bias, Ftype(1), top_data);
  }
}

template <typename Ftype, typename Btype>
void EmbedLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  CHECK(!propagate_down[0]) << "Can't backpropagate to EmbedLayer input.";
  if (this->param_propagate_down_[0]) {
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    const Btype* bottom_data = bottom[0]->cpu_data<Btype>();
    // Gradient with respect to weight
    Btype* weight_diff = this->blobs_[0]->template mutable_cpu_diff<Btype>();
    int index;
    for (int n = 0; n < M_; ++n) {
      index = static_cast<int>(bottom_data[n]);
      DCHECK_GE(index, 0);
      DCHECK_LT(index, K_);
      DCHECK_EQ(static_cast<Btype>(index), bottom_data[n])
          << "non-integer input";
      caffe_axpy(N_, Btype(1), top_diff + n * N_, weight_diff + index * N_);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    Btype* bias_diff = this->blobs_[1]->template mutable_cpu_diff<Btype>();
    caffe_cpu_gemv(CblasTrans, M_, N_, Btype(1), top_diff,
        bias_multiplier_.template cpu_data<Btype>(), Btype(1), bias_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(EmbedLayer);
#endif

INSTANTIATE_CLASS_FB(EmbedLayer);
REGISTER_LAYER_CLASS(Embed);

}  // namespace caffe
