#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmax_layer.hpp"

#ifdef USE_CUDNN

#include "caffe/layers/cudnn_softmax_layer.hpp"

#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class SoftmaxLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaxLayerTest() : blob_bottom_(new TBlob<Dtype>(2, 10, 2, 3)), blob_top_(new TBlob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~SoftmaxLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  TBlob<Dtype>* const blob_bottom_;
  TBlob<Dtype>* const blob_top_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  SoftmaxLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        Dtype sum = 0;
        for (int j = 0; j < this->blob_top_->channels(); ++j) {
          sum += this->blob_top_->data_at(i, j, k, l);
        }
        EXPECT_GE(sum, 0.998);
        EXPECT_LE(sum, 1.002);
        // Test exact values
        Dtype scale = 0;
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          scale += exp(this->blob_bottom_->data_at(i, j, k, l));
        }
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + tol<Dtype>(1e-4, 1e-2),
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale) << "debug: " << i << " " << j;
          EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - tol<Dtype>(1e-4, 1e-2),
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale) << "debug: " << i << " " << j;
        }
      }
    }
  }
}

TYPED_TEST(SoftmaxLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  SoftmaxLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-2, 1e-1), tol<Dtype>(1e-3, 1e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

#ifdef USE_CUDNN

template<typename Dtype>
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNSoftmaxLayerTest() : blob_bottom_(new TBlob<Dtype>(2, 10, 2, 3)),
                            blob_top_(new TBlob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CuDNNSoftmaxLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  TBlob<Dtype>* const blob_bottom_;
  TBlob<Dtype>* const blob_top_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNSoftmaxLayerTest, TestDtypes);

TYPED_TEST(CuDNNSoftmaxLayerTest, TestForwardCuDNN) {
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  CuDNNSoftmaxLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        TypeParam sum = 0;
        for (int j = 0; j < this->blob_top_->channels(); ++j) {
          sum += this->blob_top_->data_at(i, j, k, l);
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        TypeParam scale = 0;
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          scale += exp(this->blob_bottom_->data_at(i, j, k, l));
        }
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + tol<TypeParam>(1e-4, 1e-2),
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale) << "debug: " << i << " " << j;
          EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - tol<TypeParam>(1e-4, 1e-2),
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale) << "debug: " << i << " " << j;
        }
      }
    }
  }
}

TYPED_TEST(CuDNNSoftmaxLayerTest, TestGradientCuDNN) {
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  CuDNNSoftmaxLayer<TypeParam, TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(tol<TypeParam>(1e-2, 1e-1), tol<TypeParam>(1e-3, 1e-2));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

#endif

}  // namespace caffe
