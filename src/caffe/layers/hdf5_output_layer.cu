#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void HDF5OutputLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  CHECK_GE(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  data_blob_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                     bottom[0]->height(), bottom[0]->width());
  label_blob_.Reshape(bottom[1]->num(), bottom[1]->channels(),
                     bottom[1]->height(), bottom[1]->width());
  const int data_datum_dim = bottom[0]->count() / bottom[0]->num();
  const int label_datum_dim = bottom[1]->count() / bottom[1]->num();

  for (int i = 0; i < bottom[0]->num(); ++i) {
    caffe_copy(data_datum_dim, &bottom[0]->gpu_data<Ftype>()[i * data_datum_dim],
        &data_blob_.mutable_cpu_data()[i * data_datum_dim]);
    caffe_copy(label_datum_dim, &bottom[1]->gpu_data<Ftype>()[i * label_datum_dim],
        &label_blob_.mutable_cpu_data()[i * label_datum_dim]);
  }
  SaveBlobs();
}

template <typename Ftype, typename Btype>
void HDF5OutputLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(HDF5OutputLayer);

}  // namespace caffe
