#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::unique_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxWithLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxWithLossLayerTest()
      : blob_bottom_data_(new TBlob<Dtype>(10, 5, 2, 3)),
        blob_bottom_label_(new TBlob<Dtype>(10, 1, 2, 3)),
        blob_top_loss_(new TBlob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(tol<Dtype>(10., 0.3));
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SoftmaxWithLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  TBlob<Dtype>* const blob_bottom_data_;
  TBlob<Dtype>* const blob_bottom_label_;
  TBlob<Dtype>* const blob_top_loss_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxWithLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  if (!is_precise<Dtype>()) {
    return;
  }
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxWithLossLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-2, 1e-1), tol<Dtype>(1e-3, 1e-2));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  // First, compute the loss with all labels
  unique_ptr<SoftmaxWithLossLayer<Dtype, Dtype> > layer(
      new SoftmaxWithLossLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
  // Now, accumulate the loss, ignoring each label in {0, ..., 4} in turn.
  Dtype accum_loss = 0.F;
  for (int label = 0; label < 5; ++label) {
    layer_param.mutable_loss_param()->set_ignore_label(label);
    layer.reset(new SoftmaxWithLossLayer<Dtype, Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    accum_loss += this->blob_top_loss_->cpu_data()[0];
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(4. * full_loss, accum_loss, tol<Dtype>(1e-4, 1e-1));
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradientIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  if (!is_precise<Dtype>()) {
    return;
  }
  LayerParameter layer_param;
  // labels are in {0, ..., 4}, so we'll ignore about a fifth of them
  layer_param.mutable_loss_param()->set_ignore_label(0);
  SoftmaxWithLossLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-2, 1e-1), tol<Dtype>(1e-3, 1e-2));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  if (!is_precise<Dtype>()) {
    return;
  }
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  SoftmaxWithLossLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(5e-2, 1e-1), tol<Dtype>(5e-3, 1e-2));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
