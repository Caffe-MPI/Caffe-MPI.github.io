#ifndef CAFFE_HDF5_OUTPUT_LAYER_HPP_
#define CAFFE_HDF5_OUTPUT_LAYER_HPP_

#include "hdf5.h"

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"

/**
 * @brief Write blobs to disk as HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Ftype, typename Btype>
class HDF5OutputLayer : public Layer<Ftype, Btype> {
 public:
  explicit HDF5OutputLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param), file_opened_(false) {}
  virtual ~HDF5OutputLayer();
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {}

  virtual inline const char* type() const { return "HDF5Output"; }
  // TODO: no limit on the number of blobs
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

  inline std::string file_name() const { return file_name_; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void SaveBlobs();

  bool file_opened_;
  std::string file_name_;
  hid_t file_id_;
  TBlob<Ftype> data_blob_;
  TBlob<Ftype> label_blob_;
};

}  // namespace caffe

#endif  // CAFFE_HDF5_OUTPUT_LAYER_HPP_
