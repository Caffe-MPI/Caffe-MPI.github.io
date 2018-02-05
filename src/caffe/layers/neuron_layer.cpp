#include <vector>

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void NeuronLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS_FB(NeuronLayer);

}  // namespace caffe
