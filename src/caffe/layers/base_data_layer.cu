#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  // Note: this function runs in one thread per object and one object per one Solver thread
  shared_ptr<Batch<Ftype>> batch =
      prefetches_full_[next_batch_queue_]->pop("Data layer prefetch queue empty");
  Blob& transformed_blob = is_gpu_transform() ? *batch->gpu_transformed_data_ : batch->data_;
  if (this->relative_iter() > 1 && top[0]->data_type() == transformed_blob.data_type()
      && top[0]->shape() == transformed_blob.shape()) {
    top[0]->Swap(transformed_blob);
  } else {
    top[0]->CopyDataFrom(transformed_blob, true);
  }
  if (this->output_labels_) {
    if (this->relative_iter() > 1 && top[1]->data_type() == batch->label_.data_type()
        && top[1]->shape() == batch->label_.shape()) {
      top[1]->Swap(batch->label_);
    } else {
      top[1]->CopyDataFrom(batch->label_, true);
    }
  }
  batch->set_id((size_t) -1);
  prefetches_free_[next_batch_queue_]->push(batch);
  next_batch_queue();
}

INSTANTIATE_LAYER_GPU_FORWARD_ONLY_FB(BasePrefetchingDataLayer);

}  // namespace caffe
