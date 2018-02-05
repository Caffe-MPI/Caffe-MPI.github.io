#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <boost/make_shared.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/tensor.hpp"
#include "caffe/type.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

const int kMaxBlobAxes = 32;

namespace caffe {

template<typename Dtype>
class TBlob;

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 *        This is template-less implementation made for mixed precision.
 *        Instances can be converted to any other supported Type.
 *
 * TODO(dox): more thorough description.
 */
class Blob {
 public:
  void Swap(Blob& other) noexcept {
    std::swap(data_tensor_, other.data_tensor_);
    std::swap(diff_tensor_, other.diff_tensor_);
    std::swap(shape_data_, other.shape_data_);
    std::swap(shape_, other.shape_);
    std::swap(count_, other.count_);
    std::swap(last_data_type_, other.last_data_type_);
    std::swap(last_diff_type_, other.last_diff_type_);
  }

 protected:
  Blob(Type data_type, Type diff_type)
      : data_tensor_(make_shared<Tensor>(data_type)),
        diff_tensor_(make_shared<Tensor>(diff_type)),
        count_(0), last_data_type_(data_type), last_diff_type_(diff_type) {}
  explicit Blob(Type dtype)
      : Blob(dtype, dtype) {}

 public:
  virtual ~Blob() {}

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height, const int width);
  void Reshape(const int num);

  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);

  void ReshapeLike(const Blob& other) {
    Reshape(other.shape());
  }

  void ReshapeLike(const Blob* other) {
    Reshape(other->shape());
  }

  Type data_type() const {
    return data_tensor_ ? data_tensor_->type() : last_data_type_;
  }

  Type diff_type() const {
    return diff_tensor_ ? diff_tensor_->type() : last_diff_type_;
  }

  void lock_data() {
    data_tensor_->lock_tensor();
  }

  void lock_diff() {
    diff_tensor_->lock_tensor();
  }

  bool diff_equals(const Blob& other) const {
    return diff_tensor_ == other.diff_tensor_;
  }

  void allocate_data(bool on_gpu = true) {
    data_tensor_->mutable_memory(data_tensor_->type(), on_gpu);
  }

  void allocate_diff(bool on_gpu = true) {
    diff_tensor_->mutable_memory(diff_tensor_->type(), on_gpu);
  }

  size_t cpu_memory_data_use() const;
  size_t cpu_memory_diff_use() const;

  /**
   * @brief Creates an instance of a Blob with given Dtype.
   */
  template<typename D, typename DI = D>
  static shared_ptr<Blob> create() {
    return shared_ptr<Blob>(new Blob(tp<D>(), tp<DI>()));
  }

  /**
   * @brief Creates an instance of a Blob with given Type.
   */
  static shared_ptr<Blob> create(Type data_type, Type diff_type) {
    return shared_ptr<Blob>(new Blob(data_type, diff_type));
  }

  /// @brief Creates an instance of a Blob with given type Dtype and given shape.
  template<typename D, typename DI = D>
  static shared_ptr<Blob> create(const vector<int>& shape) {
    shared_ptr<Blob> ptr = create<D, DI>();
    ptr->Reshape(shape);
    return ptr;
  }

  template<typename D, typename DI = D>
  static shared_ptr<Blob> create(int N) {
    shared_ptr<Blob> ptr = create<D, DI>();
    vector<int> shape;
    shape.push_back(N);
    ptr->Reshape(shape);
    return ptr;
  }


  /// @brief Deprecated; use <code>create(const vector<int>& shape)</code>.
  template<typename D, typename DI = D>
  static shared_ptr<Blob> create(int num, int channels, int height, int width) {
    shared_ptr<Blob> ptr = create<D, DI>();
    ptr->Reshape(num, channels, height, width);
    return ptr;
  }

  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob& source, bool copy_diff = false, bool reshape = false);

  void CopyDataFrom(const Blob& source, bool reshape = false) {
    CopyFrom(source, false, reshape);
  }

  void CopyDiffFrom(const Blob& source, bool reshape = false) {
    CopyFrom(source, true, reshape);
  }

  bool is_data_empty() const {
    return data_tensor_->is_empty();
  }

  bool is_diff_empty() const {
    return diff_tensor_->is_empty();
  }

  std::string shape_string() const {
    std::ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }

  const vector<int>& shape() const { return shape_; }

  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }

  int num_axes() const { return shape_.size(); }
  int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }

  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes()) << "axis " << axis_index << " out of range for " << num_axes()
                                      << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes()) << "axis " << axis_index << " out of range for " << num_axes()
                                     << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  int num() const { return LegacyShape(0); }

  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  int channels() const { return LegacyShape(1); }

  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  int height() const { return LegacyShape(2); }

  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  int width() const { return LegacyShape(3); }

  int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4) << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }

  size_t offset(size_t n, size_t c = 0, size_t h = 0, size_t w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

  template<typename Dtype>
  void set_cpu_data(Dtype* data) {
    CHECK_NOTNULL(data);
    convert_data(tp<Dtype>());
    CHECK(is_type<Dtype>(data_type()));
    data_tensor_->mutable_synced_mem()->set_cpu_data(data);
  }

  template<typename Dtype>
  void set_cpu_diff(Dtype* diff) {
    CHECK_NOTNULL(diff);
    convert_diff(tp<Dtype>());
    CHECK(is_type<Dtype>(diff_type()));
    diff_tensor_->mutable_synced_mem()->set_cpu_data(diff);
  }

  template<typename Dtype>
  const Dtype* cpu_data() const {
    convert_data(tp<Dtype>());
    return static_cast<const Dtype*>(data_tensor_->synced_mem()->cpu_data());
  }

  template<typename Dtype>
  const Dtype* cpu_diff() const {
    convert_diff(tp<Dtype>());
    return static_cast<const Dtype*>(diff_tensor_->synced_mem()->cpu_data());
  }

  template<typename Dtype>
  Dtype* mutable_cpu_data() {
    convert_data(tp<Dtype>());
    return static_cast<Dtype*>(data_tensor_->mutable_synced_mem()->mutable_cpu_data());
  }

  template<typename Dtype>
  Dtype* mutable_cpu_diff() {
    convert_diff(tp<Dtype>());
    return static_cast<Dtype*>(diff_tensor_->mutable_synced_mem()->mutable_cpu_data());
  }

  // Element-wise accessor. Might be slow due to syncing from GPU to CPU.
  // Currently it's used in tests only. We better keep it this way.
  float data_at(const int n, const int c, const int h, const int w) const {
    return at(offset(n, c, h, w), data_type(), data_tensor_->synced_mem()->cpu_data());
  }
  float diff_at(const int n, const int c, const int h, const int w) const {
    return at(offset(n, c, h, w), diff_type(), diff_tensor_->synced_mem()->cpu_data());
  }
  float data_at(const vector<int>& index) const {
    return at(offset(index), data_type(), data_tensor_->synced_mem()->cpu_data());
  }
  float diff_at(const vector<int>& index) const {
    return at(offset(index), diff_type(), diff_tensor_->synced_mem()->cpu_data());
  }
  float data_at(int index) const {
    return at(index, data_type(), data_tensor_->synced_mem()->cpu_data());
  }
  float diff_at(int index) const {
    return at(index, diff_type(), diff_tensor_->synced_mem()->cpu_data());
  }

  void Update();

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  float asum_data() const {
    return data_tensor_->asum();
  }

  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  float asum_diff() const {
    return diff_tensor_->asum();
  }

  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  float sumsq_data() const;

  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  float sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  void scale_data(float scale, void* handle = nullptr, bool synced = true) {
    data_tensor_->scale(scale, handle, synced);
  }

  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(float scale, void* handle = nullptr, bool synced = true) {
    diff_tensor_->scale(scale, handle, synced);
  }

#ifndef CPU_ONLY
  /// @brief Scale the blob data by a constant factor. Uses GPU, may be asynchronous
  void gpu_scale_data(float scale, cublasHandle_t cublas_handle, bool synced) {
    data_tensor_->gpu_scale(scale, cublas_handle, synced);
  }

  /// @brief Scale the blob diff by a constant factor. Uses GPU, may be asynchronous
  void gpu_scale_diff(float scale, cublasHandle_t cublas_handle, bool synced) {
    diff_tensor_->gpu_scale(scale, cublas_handle, synced);
  }
#endif

  /// @brief Set all the blob's data elements to a value.
  void set_data(float value) {
    data_tensor_->set(value);
  }

  /// @brief Set all the blob's diff elements to a value.
  void set_diff(float value) {
    diff_tensor_->set(value);
  }

  // these are "pure algorithmic" aggregates, i.e. they depend on data array only.
  // so, "math amax" = "pure array-based amax" * "scale_"
  float amax_data() const;
  float amax_diff() const;

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);

  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);


  template<typename Dtype>
  void ToProto(BlobProto* proto, bool write_diff = false) const;
  void FromProto(const BlobProto& proto, bool reshape = true);
  bool ShapeEquals(const BlobProto& other);
  std::string to_string(int indent = 0) const;  // debug helper

  // These ones are to be used with care: they don't convert.
  void* current_mutable_data_memory(bool is_gpu) {
    return data_tensor_->current_mutable_memory(is_gpu);
  }

  void* current_mutable_diff_memory(bool is_gpu) {
    return diff_tensor_->current_mutable_memory(is_gpu);
  }

  const void* current_data_memory(bool is_gpu) {
    return data_tensor_->current_memory(is_gpu);
  }

  const void* current_diff_memory(bool is_gpu) {
    return diff_tensor_->current_memory(is_gpu);
  }

#ifndef CPU_ONLY
  size_t gpu_memory_data_use() const;
  size_t gpu_memory_diff_use() const;

  void set_gpu_data(void* data) {
    CHECK_NOTNULL(data);
    data_tensor_->mutable_synced_mem()->set_gpu_data(data);
  }

  void set_gpu_diff(void* diff) {
    CHECK_NOTNULL(diff);
    diff_tensor_->mutable_synced_mem()->set_gpu_data(diff);
  }

  template<typename Dtype>
  void set_gpu_data(Dtype* data) {
    CHECK_NOTNULL(data);
    convert_data(tp<Dtype>());
    CHECK(is_type<Dtype>(data_type()));
    data_tensor_->mutable_synced_mem()->set_gpu_data(data);
  }

  template<typename Dtype>
  void set_gpu_diff(Dtype* diff) {
    CHECK_NOTNULL(diff);
    convert_diff(tp<Dtype>());
    CHECK(is_type<Dtype>(diff_type()));
    diff_tensor_->mutable_synced_mem()->set_gpu_data(diff);
  }

  template<typename Dtype>
  const Dtype* gpu_data() const {
    convert_data(tp<Dtype>());
    return static_cast<const Dtype*>(data_tensor_->synced_mem()->gpu_data());
  }

  template<typename Dtype>
  const Dtype* gpu_diff() const {
    convert_diff(tp<Dtype>());
    return static_cast<const Dtype*>(diff_tensor_->synced_mem()->gpu_data());
  }

  template<typename Dtype>
  Dtype* mutable_gpu_data() {
    convert_data(tp<Dtype>());
    return static_cast<Dtype*>(data_tensor_->mutable_synced_mem()->mutable_gpu_data());
  }

  template<typename Dtype>
  Dtype* mutable_gpu_diff() {
    convert_diff(tp<Dtype>());
    return static_cast<Dtype*>(diff_tensor_->mutable_synced_mem()->mutable_gpu_data());
  }

  void async_gpu_push() {
    data_tensor_->mutable_synced_mem()->async_gpu_push();
  }

  const int* gpu_shape() const;
#endif

 protected:
  mutable shared_ptr<Tensor> data_tensor_;
  mutable shared_ptr<Tensor> diff_tensor_;
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;
  int count_;
  std::mutex reshape_mutex_;
  Type last_data_type_, last_diff_type_; // in case of move

  bool is_current_data_valid() const {
    return data_tensor_->is_current_valid();
  }

  bool is_current_diff_valid() const {
    return diff_tensor_->is_current_valid();
  }

  void convert_data(Type new_data_type) const {
    data_tensor_->convert(new_data_type);
  }

  void convert_diff(Type new_diff_type) const {
    diff_tensor_->convert(new_diff_type);
  }

  static float at(int offset, Type dtype, const void* data);
  static float cpu_sumsq(int count, Type dtype, const void* data);
  static void cpu_axpy(int count, Type dtype, float alpha, const void* X, void* Y);

#ifndef CPU_ONLY
  static float gpu_sumsq(int count, Type dtype, const void* data);
  static void gpu_axpy(int count, Type dtype, float alpha, const void* X, void* Y);
#endif

  static void check_integrity(bool do_data, Type current_type, Type new_type) {
    CHECK_EQ(current_type, new_type)
      << "Attempt to change TBlob native " << (do_data ? "data" : "diff")
      << " type from " << Type_Name(current_type) << " to " << Type_Name(new_type);
  }

 private:
  // Element-wise mutator. Might be slow due to syncing from GPU to CPU.
  // Currently it's used in copying from proto only.
  template<typename Dtype>
  void set_value_at(bool set_data, int idx, Dtype val) {
    void* ptr = set_data ? current_mutable_data_memory(false) : current_mutable_diff_memory(false);
    CHECK_NOTNULL(ptr);
    Type dtype = set_data ? data_type() : diff_type();
    if (is_type<float>(dtype)) {
      static_cast<float*>(ptr)[idx] = val;
    }
#ifndef CPU_ONLY
    else if (is_type<float16>(dtype)) {
      static_cast<float16*>(ptr)[idx] = val;
    }
#endif
    else if (is_type<double>(dtype)) {
      static_cast<double*>(ptr)[idx] = val;
    } else {
      LOG(FATAL) << "Unknown data or diff: " << Type_Name(dtype);
    }
  }

  DISABLE_COPY_MOVE_AND_ASSIGN(Blob);
};  // class Blob


/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 *        This is template implementation made for simpler instantiation.
 *        Instances can be converted to any other supported Type.
 *
 * TODO(dox): more thorough description.
 */
template<typename Dtype>
class TBlob : public Blob {
 public:
  TBlob()
      : Blob(tp<Dtype>()) {}

  /// @brief Deprecated; use <code>TBlob(const vector<int>& shape)</code>.
  TBlob(const int num, const int channels, const int height, const int width)
      : Blob(tp<Dtype>()) {
    Reshape(num, channels, height, width);
  }

  explicit TBlob(const vector<int>& shape)
      : Blob(tp<Dtype>()) {
    Reshape(shape);
  }

  // Shadowing parent's implementations and calling them with Dtype by default.
  // We might get rid of shadowing but overall changes would be too dramatic
  // in this case. These are the shortcuts allowing to keep current
  // code intact (pretty much).
  template<typename T = Dtype>
  const T* cpu_data() const {
    check_integrity(true, data_type(), tp<T>());
    return Blob::cpu_data<T>();
  }

  template<typename T = Dtype>
  const T* cpu_diff() const {
    check_integrity(false, diff_type(), tp<T>());
    return Blob::cpu_diff<T>();
  }

  template<typename T = Dtype>
  T* mutable_cpu_data() {
    check_integrity(true, data_type(), tp<T>());
    return Blob::mutable_cpu_data<T>();
  }

  template<typename T = Dtype>
  T* mutable_cpu_diff() {
    check_integrity(false, diff_type(), tp<T>());
    return Blob::mutable_cpu_diff<T>();
  }

#ifndef CPU_ONLY
  template<typename T = Dtype>
  const T* gpu_data() const {
    check_integrity(true, data_type(), tp<T>());
    return Blob::gpu_data<T>();
  }

  template<typename T = Dtype>
  const T* gpu_diff() const {
    check_integrity(false, diff_type(), tp<T>());
    return Blob::gpu_diff<T>();
  }

  template<typename T = Dtype>
  T* mutable_gpu_data() {
    check_integrity(true, data_type(), tp<T>());
    return Blob::mutable_gpu_data<T>();
  }

  template<typename T = Dtype>
  T* mutable_gpu_diff() {
    check_integrity(false, diff_type(), tp<T>());
    return Blob::mutable_gpu_diff<T>();
  }
#endif

  DISABLE_COPY_MOVE_AND_ASSIGN(TBlob);
};  // class TBlob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
