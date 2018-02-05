#include <stdint.h>  // for uint32_t & uint64_t
#include <time.h>
#include <climits>
#include <cmath>  // for std::fabs
#include <cstdlib>  // for rand_r

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MathFunctionsTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MathFunctionsTest()
      : blob_bottom_(new TBlob<Dtype>()),
        blob_top_(new TBlob<Dtype>()) {
  }

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    this->blob_bottom_->Reshape(11, 17, 19, 23);
    this->blob_top_->Reshape(11, 17, 19, 23);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_top_);
  }

  virtual ~MathFunctionsTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  // http://en.wikipedia.org/wiki/Hamming_distance
  int ReferenceHammingDistance(const int n, const Dtype* x, const Dtype* y) {
    int dist = 0;
    uint64_t val;
    for (int i = 0; i < n; ++i) {
      if (is_type<Dtype>(DOUBLE)) {
        val = static_cast<uint64_t>(x[i]) ^ static_cast<uint64_t>(y[i]);
      } else if (is_type<Dtype>(FLOAT)) {
        val = static_cast<uint32_t>(x[i]) ^ static_cast<uint32_t>(y[i]);
      }

#ifndef CPU_ONLY
      else if (is_type<Dtype>(FLOAT16)) {
        val = reinterpret_cast<const float16*>(x + i)->getx() ^
            reinterpret_cast<const float16*>(y + i)->getx();
      }
#endif
      // NOLINT_NEXT_LINE(readability/braces)
      else {
        LOG(FATAL) << "Unrecognized Dtype size: " << sizeof(Dtype);
      }
      // Count the number of set bits
      while (val) {
        ++dist;
        val &= val - 1;
      }
    }
    return dist;
  }

  TBlob<Dtype>* const blob_bottom_;
  TBlob<Dtype>* const blob_top_;
};

template <typename Dtype>
class CPUMathFunctionsTest
  : public MathFunctionsTest<CPUDevice<Dtype> > {
};

TYPED_TEST_CASE(CPUMathFunctionsTest, TestDtypes);

TYPED_TEST(CPUMathFunctionsTest, TestNothing) {
  // The first test case of a test suite takes the longest time
  //   due to the set up overhead.
}

TYPED_TEST(CPUMathFunctionsTest, TestAlign) {
  EXPECT_EQ(0UL, align_up<0>(0UL));
  EXPECT_EQ(0UL, align_up<1>(0UL));
  EXPECT_EQ(0UL, align_up<2>(0UL));
  EXPECT_EQ(0UL, align_up<3>(0UL));
  EXPECT_EQ(1UL, align_up<0>(1UL));
  EXPECT_EQ(2UL, align_up<1>(1UL));
  EXPECT_EQ(4UL, align_up<2>(1UL));
  EXPECT_EQ(8UL, align_up<3>(1UL));
  EXPECT_EQ(2UL, align_up<0>(2UL));
  EXPECT_EQ(2UL, align_up<1>(2UL));
  EXPECT_EQ(4UL, align_up<2>(2UL));
  EXPECT_EQ(8UL, align_up<3>(2UL));

  EXPECT_EQ(0UL, even(0UL));
  EXPECT_EQ(2UL, even(1UL));
  EXPECT_EQ(2UL, even(2UL));
  EXPECT_EQ(4UL, even(3UL));
  EXPECT_EQ(4UL, even(4UL));
  EXPECT_EQ(6UL, even(5UL));

  EXPECT_EQ(0UL, align_down<0>(0UL));
  EXPECT_EQ(0UL, align_down<1>(0UL));
  EXPECT_EQ(0UL, align_down<2>(0UL));
  EXPECT_EQ(0UL, align_down<3>(0UL));
  EXPECT_EQ(1UL, align_down<0>(1UL));
  EXPECT_EQ(0UL, align_down<1>(1UL));
  EXPECT_EQ(0UL, align_down<2>(1UL));
  EXPECT_EQ(0UL, align_down<3>(1UL));
  EXPECT_EQ(2UL, align_down<0>(2UL));
  EXPECT_EQ(2UL, align_down<1>(2UL));
  EXPECT_EQ(0UL, align_down<2>(2UL));
  EXPECT_EQ(0UL, align_down<3>(2UL));
  EXPECT_EQ(17UL, align_down<0>(17UL));
  EXPECT_EQ(16UL, align_down<1>(17UL));
  EXPECT_EQ(16UL, align_down<2>(17UL));
  EXPECT_EQ(16UL, align_down<3>(17UL));
  EXPECT_EQ(16UL, align_down<4>(17UL));
  EXPECT_EQ(0UL, align_down<5>(17UL));
}

TYPED_TEST(CPUMathFunctionsTest, TestHammingDistance) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  const TypeParam* y = this->blob_top_->cpu_data();
  int cpu_distance = caffe_cpu_hamming_distance(n, x, y);
  EXPECT_EQ(this->ReferenceHammingDistance(n, x, y),
            cpu_distance);
}

TYPED_TEST(CPUMathFunctionsTest, TestAsum) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  double std_asum = 0.;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam cpu_asum = caffe_cpu_asum<TypeParam>(n, x);
  EXPECT_LT((cpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(CPUMathFunctionsTest, TestSign) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sign<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestSgnbit) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sgnbit<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestFabs) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(abs_val[i], x[i] > 0 ? x[i] : -x[i]);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestScale) {
  int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  caffe_cpu_scale<TypeParam>(n, alpha, this->blob_bottom_->cpu_data(),
                             this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* scaled = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(scaled[i], x[i] * alpha);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestCopy) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  TypeParam* top_data = this->blob_top_->mutable_cpu_data();
  caffe_copy(n, bottom_data, top_data);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

#ifndef CPU_ONLY

template <typename Dtype>
class GPUMathFunctionsTest : public MathFunctionsTest<GPUDevice<Dtype> > {
};

TYPED_TEST_CASE(GPUMathFunctionsTest, TestDtypes);

  //TODO
  //  TYPED_TEST(GPUMathFunctionsTest, TestHammingDistance) {
  //  int n = this->blob_bottom_->count();
  //  const TypeParam* x = this->blob_bottom_->cpu_data();
  //  const TypeParam* y = this->blob_top_->cpu_data();
  //  int reference_distance = this->ReferenceHammingDistance(n, x, y);
  //  x = this->blob_bottom_->gpu_data();
  //  y = this->blob_top_->gpu_data();
  //  int computed_distance = caffe_gpu_hamming_distance(n, x, y);
  //  EXPECT_EQ(reference_distance, computed_distance);
  //  }

TYPED_TEST(GPUMathFunctionsTest, TestAsum) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  double std_asum = 0.;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam gpu_asum;
  caffe_gpu_asum(n, this->blob_bottom_->gpu_data(), &gpu_asum);
  EXPECT_LT((gpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(GPUMathFunctionsTest, TestAmax) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam m, std_amax = 0.;
  int std_nnz = 0;
  float std_asum = 0.;
  for (int i = 0; i < n; ++i) {
    m = std::fabs(x[i]);
    std_asum += m;
    if (m > std_amax) {
      std_amax = m;
    }
    if (x[i] != 0) {
      std_nnz++;
    }
  }
  float gpu_amax;
  caffe_gpu_amax(n, this->blob_bottom_->gpu_data(), &gpu_amax);
  EXPECT_FLOAT_EQ(static_cast<float>(std_amax), static_cast<float>(gpu_amax));

  // pow2 test
  for (int j = 0; j < 22; ++j) {
    Caffe::set_random_seed(1391);
    TBlob<TypeParam> b;
    b.Reshape(1 << j, 1, 1, 1);
    FillerParameter filler_param;
    GaussianFiller<TypeParam> filler(filler_param);
    filler.Fill(&b);
    n = b.count();
    x = b.cpu_data();
    std_amax = 0.;
    std_asum = 0.F;
    std_nnz = 0;
    for (int i = 0; i < n; ++i) {
      m = std::fabs(x[i]);
      if (m > std_amax) {
        std_amax = m;
      }
      std_asum += m;
      if (m > std_amax) {
        std_amax = m;
      }
      if (x[i] != 0) {
        std_nnz++;
      }
    }
    gpu_amax = b.amax_data();
    EXPECT_FLOAT_EQ(static_cast<float>(std_amax), static_cast<float>(gpu_amax));
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestExtFP16) {
  if (!is_type<TypeParam>(FLOAT)) {
    return;
  }

  TBlob<float> blob, blob2;
  blob.Reshape(11, 17, 19, 23);
  blob2.Reshape(11, 17, 19, 23);
  float* data = blob.mutable_cpu_data();
  int n = blob.count();
  caffe_rng_gaussian(n, 100.F, 10.F, data);
  const float* xgpu = blob.gpu_data();

  TBlob<float16> blob16;
  blob16.Reshape(11, 17, 19, 23);
  float16* fp16gpu = blob16.mutable_gpu_data();

  caffe_gpu_convert(n, xgpu, fp16gpu);

  const float* xcpu = blob.cpu_data();
  const float16* fp16cpu = blob16.cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(xcpu[i], static_cast<float>(fp16cpu[i]), 1.e-1);
  }

  float* x2gpu = blob2.mutable_gpu_data();
  caffe_gpu_convert(n, fp16gpu, x2gpu);
  const float* x2cpu = blob2.cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(xcpu[i], x2cpu[i], 1.e-1);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestSign) {
  int n = this->blob_bottom_->count();
  caffe_gpu_sign<TypeParam>(n, this->blob_bottom_->gpu_data(),
                 this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestSgnbit) {
  int n = this->blob_bottom_->count();
  caffe_gpu_sgnbit<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestFabs) {
  int n = this->blob_bottom_->count();
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(abs_val[i], x[i] > 0 ? x[i] : -x[i]);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestScale) {
  int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  caffe_gpu_scale<TypeParam>(n, alpha, this->blob_bottom_->gpu_data(),
                             this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* scaled = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(scaled[i], x[i] * alpha);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestScaleInPlace) {
  const int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  TBlob<TypeParam> original;
  original.CopyFrom(*this->blob_bottom_, true, true);

  caffe_gpu_scal<TypeParam>(n, (TypeParam)alpha,
      this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* scaled = this->blob_bottom_->cpu_diff();
  const TypeParam* x = original.cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(scaled[i]), static_cast<float>(x[i] * alpha));
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestCopy) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();
  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  caffe_copy(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestConvertToAndFrom16) {
  const int n = this->blob_bottom_->count();
  TypeParam* bottom_data_cpu = this->blob_bottom_->mutable_cpu_data();
  bottom_data_cpu[0] = 70000.;
  bottom_data_cpu[1] = -80000.;
  TBlob<float16> b16;
  b16.ReshapeLike(this->blob_bottom_);
  float16* b16_data = b16.mutable_gpu_data();
  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();

  caffe_gpu_convert(n, bottom_data, b16_data);
  const float16* b16_data_cpu = b16.cpu_data();
  if (is_precise<TypeParam>()) {
    EXPECT_FLOAT_EQ(65504.F, static_cast<float>(b16_data_cpu[0]));
    EXPECT_FLOAT_EQ(-65504.F, static_cast<float>(b16_data_cpu[1]));
  }
  for (int i = 2; i < n; ++i) {
    EXPECT_NEAR(static_cast<float>(bottom_data_cpu[i]),
        static_cast<float>(b16_data_cpu[i]), 2.e-3) << " at i=" << i;
  }

  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  caffe_gpu_convert(n, b16_data, top_data);
  const TypeParam* top_data_cpu = this->blob_top_->cpu_data();
  if (is_precise<TypeParam>()) {
    EXPECT_FLOAT_EQ(65504.F, static_cast<float>(top_data_cpu[0]));
    EXPECT_FLOAT_EQ(-65504.F, static_cast<float>(top_data_cpu[1]));
  }
  for (int i = 2; i < n; ++i) {
    EXPECT_NEAR(static_cast<float>(bottom_data_cpu[i]),
        static_cast<float>(top_data_cpu[i]), 2.e-3) << " at i=" << i;
  }
}
#endif

}  // namespace caffe
