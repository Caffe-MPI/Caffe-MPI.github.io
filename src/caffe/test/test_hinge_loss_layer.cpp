#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/hinge_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class HingeLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HingeLossLayerTest()
      : blob_bottom_data_(new TBlob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new TBlob<Dtype>(10, 1, 1, 1)),
        blob_top_loss_(new TBlob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~HingeLossLayerTest() {
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

TYPED_TEST_CASE(HingeLossLayerTest, TestDtypesAndDevices);


TYPED_TEST(HingeLossLayerTest, TestGradientL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HingeLossLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(3e-2, 1e-1),
      tol<Dtype>(2e-3, 5e-2), 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(HingeLossLayerTest, TestGradientL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Set norm to L2
  HingeLossParameter* hinge_loss_param = layer_param.mutable_hinge_loss_param();
  hinge_loss_param->set_norm(HingeLossParameter_Norm_L2);
  HingeLossLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-2, 5e-1),
      tol<Dtype>(1e-2, 5e-1), 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
