#ifndef CAFFE_MEMORY_DATA_LAYER_HPP_
#define CAFFE_MEMORY_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Ftype, typename Btype>
class MemoryDataLayer : public BaseDataLayer<Ftype, Btype> {
 public:
  explicit MemoryDataLayer(const LayerParameter& param)
      : BaseDataLayer<Ftype, Btype>(param, 1), has_new_data_(false) {}
  virtual void DataLayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "MemoryData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  virtual void AddDatumVector(const vector<Datum>& datum_vector);
#ifdef USE_OPENCV
  virtual void AddMatVector(const vector<cv::Mat>& mat_vector,
      const vector<int>& labels);
#endif  // USE_OPENCV

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to TBlob, which is mutable
  void Reset(Ftype* data, Ftype* label, int n);
  void set_batch_size(int new_size);

  int batch_size() { return batch_size_; }
  int channels() { return channels_; }
  int height() { return height_; }
  int width() { return width_; }

 protected:
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override {
    Forward_cpu(bottom, top);
  }

  int batch_size_, channels_, height_, width_, size_;
  Ftype* data_;
  Ftype* labels_;
  int n_;
  size_t pos_;
  TBlob<Ftype> added_data_;
  TBlob<Ftype> added_label_;
  bool has_new_data_;
};

}  // namespace caffe

#endif  // CAFFE_MEMORY_DATA_LAYER_HPP_
