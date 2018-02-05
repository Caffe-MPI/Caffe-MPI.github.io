#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/batch_reindex_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class BatchReindexLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BatchReindexLayerTest()
      : blob_bottom_(new TBlob<Dtype>()),
        blob_bottom_permute_(new TBlob<Dtype>()),
        blob_top_(new TBlob<Dtype>()) {
  }
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    vector<int> sz;
    sz.push_back(5);
    sz.push_back(4);
    sz.push_back(3);
    sz.push_back(2);
    blob_bottom_->Reshape(sz);
    vector<int> permsz;
    permsz.push_back(6);
    blob_bottom_permute_->Reshape(permsz);

    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    int perm[] = { 4, 0, 4, 0, 1, 2 };
    for (int i = 0; i < blob_bottom_permute_->count(); ++i) {
      blob_bottom_permute_->mutable_cpu_data()[i] = perm[i];
    }

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_permute_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~BatchReindexLayerTest() {
    delete blob_bottom_permute_;
    delete blob_bottom_;
    delete blob_top_;
  }
  TBlob<Dtype>* const blob_bottom_;
  TBlob<Dtype>* const blob_bottom_permute_;
  TBlob<Dtype>* const blob_top_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;

  void TestForward() {
    LayerParameter layer_param;

    vector<int> sz;
    sz.push_back(5);
    sz.push_back(4);
    sz.push_back(3);
    sz.push_back(2);
    blob_bottom_->Reshape(sz);
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      blob_bottom_->mutable_cpu_data()[i] = i;
    }

    vector<int> permsz;
    permsz.push_back(6);
    blob_bottom_permute_->Reshape(permsz);
    int perm[] = { 4, 0, 4, 0, 1, 2 };
    for (int i = 0; i < blob_bottom_permute_->count(); ++i) {
      blob_bottom_permute_->mutable_cpu_data()[i] = perm[i];
    }
    BatchReindexLayer<Dtype, Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), blob_bottom_permute_->num());
    EXPECT_EQ(blob_top_->channels(), blob_bottom_->channels());
    EXPECT_EQ(blob_top_->height(), blob_bottom_->height());
    EXPECT_EQ(blob_top_->width(), blob_bottom_->width());

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    int channels = blob_top_->channels();
    int height = blob_top_->height();
    int width = blob_top_->width();
    for (int i = 0; i < blob_top_->count(); ++i) {
      int n = i / (channels * width * height);
      int inner_idx = (i % (channels * width * height));
      EXPECT_EQ(
          blob_top_->cpu_data()[i],
          blob_bottom_->cpu_data()[perm[n] * channels * width * height
              + inner_idx]);
    }
  }
};

TYPED_TEST_CASE(BatchReindexLayerTest, TestDtypesAndDevices);

TYPED_TEST(BatchReindexLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(BatchReindexLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BatchReindexLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-4, 1e-2), tol<Dtype>(1e-2, 1e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  }

}  // namespace caffe
