#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_
#ifdef USE_CUDNN

#include <cudnn.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/float16.hpp"

#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#if !defined(CUDNN_VERSION) || !CUDNN_VERSION_MIN(6, 0, 0)
#error "NVCaffe 0.16 and higher requires CuDNN version 6.0.0 or higher"
#endif

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "\
      << cudnnGetErrorString(status); \
  } while (0)

const char* cudnnGetErrorString(cudnnStatus_t status);

namespace caffe {

namespace cudnn {

template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static const cudnnDataType_t conv_type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static const cudnnDataType_t conv_type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};
#ifndef CPU_ONLY
template<> class dataType<float16> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_HALF;
  static const cudnnDataType_t conv_type = CUDNN_DATA_HALF;
  static float oneval, zeroval;
  static const void *one, *zero;
};
#endif

inline
cudnnDataType_t conv_type(Type math) {
  cudnnDataType_t ret;
  switch (math) {
  case FLOAT:
    ret = dataType<float>::conv_type;
    break;
#ifndef CPU_ONLY
  case FLOAT16:
    // TODO
    if (caffe::Caffe::device_capability(caffe::Caffe::current_device()) == 600) {
      ret = dataType<float16>::conv_type;
    } else {
      ret = dataType<float>::conv_type;
    }
    break;
#endif
  case DOUBLE:
    ret = dataType<double>::conv_type;
    break;
  default:
    LOG(FATAL) << "Unknown Math type " << Type_Name(math);
    break;
  }
  return ret;
}

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
        n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
                         stride_n, stride_c, stride_h, stride_w);
}

}  // namespace cudnn

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_H_
