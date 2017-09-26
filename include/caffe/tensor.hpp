#ifndef INCLUDE_CAFFE_TENSOR_HPP_
#define INCLUDE_CAFFE_TENSOR_HPP_

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/type.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

class Tensor {
  friend class Blob;

 public:
  explicit Tensor(Type type);

  ~Tensor() {}

  std::string to_string(int indent) const;

  static void copy_helper(bool use_gpu, int count, const void* p_src, Type src_type,
      void* p_dst, Type dst_type);  // NOLINT(runtime/references)

#ifndef CPU_ONLY
  static void
  gpu_scal(int count, Type dtype, void* data, float scal, cublasHandle_t cublas_handle, bool sync);
#endif
  static void cpu_scal(int count, Type dtype, void* data, float scal);

 private:
  Type type() const {
    return type_;
  }

  void lock_tensor() {
    locked_ = true;
  }

  size_t size() const {
    return synced_arrays_->size();
  }

  void set(float value);
  void scale(float new_scale, void* handle = nullptr, bool synced = true);
  void cpu_scale(float new_scale);
  float cpu_amax();
  float cpu_asum();
  float asum() const;
  float sumsq() const;
  void invalidate_others();

#ifndef CPU_ONLY
  void gpu_set(float value, bool sync, cudaStream_t stream);
  void gpu_scale(float new_scale, cublasHandle_t cublas_handle, bool sync);
  float gpu_amax();
  size_t gpu_memory_use() const;
#endif
  size_t cpu_memory_use() const;
  const shared_ptr<SyncedMemory>& synced_mem() const;
  shared_ptr<SyncedMemory>& mutable_synced_mem(bool flush = true);
  void convert(Type new_type);
  void Reshape(int count);

  bool is_current_valid() const {
    const shared_ptr<SyncedMemory>& mem = synced_arrays_->at(type_);
    return mem && mem->is_valid();
  }

  void* mutable_memory(Type type, bool is_gpu) {
    convert(type);
    shared_ptr<SyncedMemory>& mem = mutable_synced_mem();
    return is_gpu ? mem->mutable_gpu_data() : mem->mutable_cpu_data();
  }

  void* current_mutable_memory(bool is_gpu) {
    shared_ptr<SyncedMemory>& mem = mutable_synced_mem();
    return is_gpu ? mem->mutable_gpu_data() : mem->mutable_cpu_data();
  }

  const void* current_memory(bool is_gpu) {
    const shared_ptr<SyncedMemory>& mem = synced_mem();
    return is_gpu ? mem->gpu_data() : mem->cpu_data();
  }

  bool is_empty() const {
    const shared_ptr<SyncedMemory>& mem = synced_arrays_->at(type_);
    return !mem || mem->head() == SyncedMemory::UNINITIALIZED;
  }

  // numerical type stored here at a moment (might change due to conversion)
  Type type_;
  bool locked_;
  // array of projections to different types (including current type_)
  shared_ptr<vector<shared_ptr<SyncedMemory>>> synced_arrays_;
  // number of entries - comes from Blob via Reshape
  int count_;

  DISABLE_COPY_MOVE_AND_ASSIGN(Tensor);
};  // class Tensor

}  // namespace caffe

#endif /* INCLUDE_CAFFE_TENSOR_HPP_ */
