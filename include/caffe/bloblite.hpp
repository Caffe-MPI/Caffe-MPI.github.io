#ifndef CAFFE_BLOBLITE_HPP_
#define CAFFE_BLOBLITE_HPP_

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Bloblite {
 public:
  Bloblite()
       : data_(), num_(0),
       count_(0), capacity_(0) {}
  explicit Bloblite(const int num);
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
  void Reshape(const int num);
  void ReshapeLike(const Bloblite& other);
  inline int num() const { return num_; }
  inline int count() const { return count_; }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */

  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const Dtype* gpu_data() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer&s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Bloblite& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer&s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */

 protected:
  shared_ptr<SyncedMemory> data_;
  int num_;
  int count_;
  int capacity_;

  DISABLE_COPY_AND_ASSIGN(Bloblite);
};  // class Blob lite

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
