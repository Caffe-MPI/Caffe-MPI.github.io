#include <vector>

#include "caffe/layers/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void SilenceLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  // Do nothing.
}

template <typename Ftype, typename Btype>
void SilenceLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_gpu_set<Btype>(bottom[i]->count(), Btype(0),
                    bottom[i]->mutable_gpu_diff<Btype>());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(SilenceLayer);

}  // namespace caffe
