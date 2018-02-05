#include <vector>

#include "caffe/layers/flatten_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void FlattenLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  const int start_axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.flatten_param().axis());
  const int end_axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.flatten_param().end_axis());
  vector<int> top_shape;
  for (int i = 0; i < start_axis; ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  const int flattened_dim = bottom[0]->count(start_axis, end_axis + 1);
  top_shape.push_back(flattened_dim);
  for (int i = end_axis + 1; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
}

template <typename Ftype, typename Btype>
void FlattenLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Ftype, typename Btype>
void FlattenLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  bottom[0]->ShareDiff(*top[0]);
}

INSTANTIATE_CLASS_FB(FlattenLayer);
REGISTER_LAYER_CLASS(Flatten);

}  // namespace caffe
