#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mvn_layer.hpp"
#include <google/protobuf/text_format.h>
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MVNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MVNLayerTest()
      : blob_bottom_(new TBlob<Dtype>(2, 3, 4, 5)),
        blob_top_(new TBlob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MVNLayerTest() { delete blob_bottom_; delete blob_top_; }
  TBlob<Dtype>* const blob_bottom_;
  TBlob<Dtype>* const blob_top_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(MVNLayerTest, TestDtypesAndDevices);

TYPED_TEST(MVNLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MVNLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      float sum = 0, var = 0;
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          float data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
      sum /= height * width;
      var /= height * width;

      const Dtype kErrorBound = tol<Dtype>(0.001, 0.002);
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
    }
  }
}

TYPED_TEST(MVNLayerTest, TestForwardMeanOnly) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param{normalize_variance: false}", &layer_param));
  MVNLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      float sum = 0, var = 0;
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          float data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
      sum /= height * width;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
    }
  }
}

TYPED_TEST(MVNLayerTest, TestForwardAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param{across_channels: true}", &layer_param));
  MVNLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    float sum = 0, var = 0;
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          float data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * channels;
    var /= height * width * channels;

    const Dtype kErrorBound = tol<Dtype>(0.001, 0.003);
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(MVNLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MVNLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-2, 6e-2), tol<Dtype>(1e-3, 1e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MVNLayerTest, TestGradientMeanOnly) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param{normalize_variance: false}", &layer_param));
  MVNLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-2, 5e-2), tol<Dtype>(1e-3, 1e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MVNLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param{across_channels: true}", &layer_param));
  MVNLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(6e-2, 1e-1), tol<Dtype>(1e-3, 1e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
