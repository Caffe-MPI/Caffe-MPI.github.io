#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void SigmoidCrossEntropyLossLayer<Ftype, Btype>::Backward_gpu(
    const vector<Blob*>& top, const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Btype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Btype* target = bottom[1]->gpu_data<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Btype(-1), target, bottom_diff);
    // Scale down gradient
    const Btype loss_weight = top[0]->cpu_diff<Btype>()[0];
    caffe_gpu_scal(count, Btype(loss_weight / num), bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD_ONLY_FB(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
