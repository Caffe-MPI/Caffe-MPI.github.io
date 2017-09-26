#include "caffe/type.hpp"
#include <glog/logging.h>
#include <climits>

namespace caffe {

size_t tsize(Type dtype) {
  switch (dtype) {
    case FLOAT:
      return sizeof(float);
    case DOUBLE:
      return sizeof(double);
#ifndef CPU_ONLY
    case FLOAT16:
      return sizeof(float16);
#endif
    case INT:
      return sizeof(int);
    case UINT:
      return sizeof(unsigned int);
    default:
      LOG(FATAL) << "Unsupported math type";
      break;
  }
  return 4;
}

}  // namespace caffe
