#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ImageDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageDataLayerTest()
      : seed_(1701),
        blob_top_data_(new TBlob<Dtype>()),
        blob_top_label_(new TBlob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << i;
    }
    outfile.close();
    // Create test input file for images of distinct sizes.
    MakeTempFilename(&filename_reshape_);
    std::ofstream reshapefile(filename_reshape_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_reshape_;
    reshapefile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << 0;
    reshapefile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg " << 1;
    reshapefile.close();
  }

  virtual ~ImageDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  string filename_reshape_;
  TBlob<Dtype>* const blob_top_data_;
  TBlob<Dtype>* const blob_top_label_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(ImageDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype, Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(5, this->blob_top_data_->num());
  EXPECT_EQ(3, this->blob_top_data_->channels());
  EXPECT_EQ(360, this->blob_top_data_->height());
  EXPECT_EQ(480, this->blob_top_data_->width());
  EXPECT_EQ(5, this->blob_top_label_->num());
  EXPECT_EQ(1, this->blob_top_label_->channels());
  EXPECT_EQ(1, this->blob_top_label_->height());
  EXPECT_EQ(1, this->blob_top_label_->width());
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, static_cast<int>(this->blob_top_label_->cpu_data()[i]));
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype, Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(5, this->blob_top_data_->num());
  EXPECT_EQ(3, this->blob_top_data_->channels());
  EXPECT_EQ(256, this->blob_top_data_->height());
  EXPECT_EQ(256, this->blob_top_data_->width());
  EXPECT_EQ(5, this->blob_top_label_->num());
  EXPECT_EQ(1, this->blob_top_label_->channels());
  EXPECT_EQ(1, this->blob_top_label_->height());
  EXPECT_EQ(1, this->blob_top_label_->width());
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, static_cast<int>(this->blob_top_label_->cpu_data()[i]));
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_reshape_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype, Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(1, this->blob_top_label_->num());
  EXPECT_EQ(1, this->blob_top_label_->channels());
  EXPECT_EQ(1, this->blob_top_label_->height());
  EXPECT_EQ(1, this->blob_top_label_->width());
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(1, this->blob_top_data_->num());
  EXPECT_EQ(3, this->blob_top_data_->channels());
  EXPECT_EQ(360, this->blob_top_data_->height());
  EXPECT_EQ(480, this->blob_top_data_->width());
  // fish-bike.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(1, this->blob_top_data_->num());
  EXPECT_EQ(3, this->blob_top_data_->channels());
  EXPECT_EQ(323, this->blob_top_data_->height());
  EXPECT_EQ(481, this->blob_top_data_->width());
}

TYPED_TEST(ImageDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(true);
  ImageDataLayer<Dtype, Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(5, this->blob_top_data_->num());
  EXPECT_EQ(3, this->blob_top_data_->channels());
  EXPECT_EQ(360, this->blob_top_data_->height());
  EXPECT_EQ(480, this->blob_top_data_->width());
  EXPECT_EQ(5, this->blob_top_label_->num());
  EXPECT_EQ(1, this->blob_top_label_->channels());
  EXPECT_EQ(1, this->blob_top_label_->height());
  EXPECT_EQ(1, this->blob_top_label_->width());
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(values_to_indices.size(), 5);
    EXPECT_GT(5, num_in_order);
  }
}

}  // namespace caffe
#endif  // USE_OPENCV
