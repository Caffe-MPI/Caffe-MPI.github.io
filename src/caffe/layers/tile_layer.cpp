#include <vector>

#include "caffe/layers/tile_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void TileLayer<Ftype, Btype>::Reshape(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const TileParameter& tile_param = this->layer_param_.tile_param();
  axis_ = bottom[0]->CanonicalAxisIndex(tile_param.axis());
  CHECK(tile_param.has_tiles()) << "Number of tiles must be specified";
  tiles_ = tile_param.tiles();
  CHECK_GT(tiles_, 0) << "Number of tiles must be positive.";
  vector<int> top_shape = bottom[0]->shape();
  top_shape[axis_] = bottom[0]->shape(axis_) * tiles_;
  top[0]->Reshape(top_shape);
  outer_dim_ = bottom[0]->count(0, axis_);
  inner_dim_ = bottom[0]->count(axis_);
}

template <typename Ftype, typename Btype>
void TileLayer<Ftype, Btype>::Forward_cpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  for (int i = 0; i < outer_dim_; ++i) {
    for (int t = 0; t < tiles_; ++t) {
      caffe_copy(inner_dim_, bottom_data, top_data);
      top_data += inner_dim_;
    }
    bottom_data += inner_dim_;
  }
}

template <typename Ftype, typename Btype>
void TileLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
  for (int i = 0; i < outer_dim_; ++i) {
    caffe_copy(inner_dim_, top_diff, bottom_diff);
    top_diff += inner_dim_;
    for (int t = 1; t < tiles_; ++t) {
      caffe_axpy(inner_dim_, Btype(1), top_diff, bottom_diff);
      top_diff += inner_dim_;
    }
    bottom_diff += inner_dim_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(TileLayer);
#endif

INSTANTIATE_CLASS_FB(TileLayer);
REGISTER_LAYER_CLASS(Tile);

}  // namespace caffe
