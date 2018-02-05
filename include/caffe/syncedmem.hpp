#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>
#include <boost/thread.hpp>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/gpu_memory.hpp"

namespace caffe {

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1), valid_(true)
      {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1), valid_(true)
      {}

  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() const { return head_; }
  size_t size() const { return size_; }
  size_t gpu_memory_use() const { return own_gpu_data_ ? size_ : 0ULL; }
  size_t cpu_memory_use() const { return own_cpu_data_ ? size_ : 0ULL; }

  bool is_valid() const {
    return valid_;
  }
  void invalidate() {
    valid_ = false;
  }
  void validate() {
    valid_ = true;
  }

  float cpu_asum(int count, Type type);
  float cpu_sumsq(int count, Type dtype);

#ifndef CPU_ONLY
  int gpu_device() const {
    return gpu_device_;
  }

  void async_gpu_push();
  float gpu_asum(int count, Type type);
  float gpu_amax(int count, Type type);
  float gpu_sumsq(int count, Type dtype);
#endif

  std::string to_string(int indent, Type type);  // debug helper

 protected:
  void MallocHost(void** ptr, size_t size, bool* use_cuda);
  void FreeHost(void* ptr, bool use_cuda);

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
  bool valid_;

  DISABLE_COPY_MOVE_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
