#ifndef CAFFE_UTIL_NCCL_H_
#define CAFFE_UTIL_NCCL_H_
#ifdef USE_NCCL
#ifndef CPU_ONLY

#include <nccl.h>

#include "caffe/common.hpp"

#define NCCL_CHECK(condition) \
{ \
  ncclResult_t result = condition; \
  CHECK_EQ(result, ncclSuccess) << " " \
    << ncclGetErrorString(result); \
}

namespace caffe {

namespace nccl {

template <typename Dtype> class dataType;

template<> class dataType<float> {
 public:
  static const ncclDataType_t type = ncclFloat;
};
template<> class dataType<double> {
 public:
  static const ncclDataType_t type = ncclDouble;
};
#ifndef CPU_ONLY
template<> class dataType<float16> {
 public:
  static const ncclDataType_t type = ncclHalf;
};
#endif

inline ncclDataType_t nccl_type(Type type) {
  ncclDataType_t ret = dataType<float>::type;
  if (is_type<float>(type)) {
    return ret;
  } else if (is_type<float16>(type)) {
    ret = dataType<float16>::type;
  } else if (is_type<double>(type)) {
    ret = dataType<double>::type;
  } else {
    LOG(FATAL) << "Type " << Type_Name(type) << " is not supported by NCCL";
  }
  return ret;
}

}  // namespace nccl

}  // namespace caffe

#endif
#endif  // end USE_NCCL
#endif  // CAFFE_UTIL_NCCL_H_
