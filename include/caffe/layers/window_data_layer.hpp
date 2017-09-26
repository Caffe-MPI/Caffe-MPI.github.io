#ifndef CAFFE_WINDOW_DATA_LAYER_HPP_
#define CAFFE_WINDOW_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Ftype, typename Btype>
class WindowDataLayer : public BasePrefetchingDataLayer<Ftype, Btype> {
 public:
  explicit WindowDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Ftype, Btype>(param) {}
  virtual ~WindowDataLayer();
  void DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  const char* type() const override { return "WindowData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }
  bool is_gpu_transform() const override { return false; }

 protected:
  unsigned int PrefetchRand();
  void load_batch(Batch<Ftype>* batch, int thread_id, size_t queue_id = 0UL) override;
  void start_reading() override {}

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
  TBlob<Ftype> data_mean_;
  vector<Ftype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
  bool cache_images_;
  vector<std::pair<std::string, Datum > > image_database_cache_;
  vector<int> data_shape_, label_shape_;
};

}  // namespace caffe

#endif  // CAFFE_WINDOW_DATA_LAYER_HPP_
