#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single/multiple thread(s),
 * by reimplementing the virtual function InternalThreadEntry/InternalThreadEntryN.
 */
class InternalThread {
 public:
  InternalThread(int target_device, size_t rank_, size_t threads, bool delayed);
  virtual ~InternalThread() {}

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread(bool set_cpu_affinity = false,
      uint64_t random_seed = Caffe::next_seed());

  /**
   * Restarts all threads
   * @param set_cpu_affinity
   */
  void RestartAllThreads(size_t new_threads, bool delayed = false, bool set_cpu_affinity = false,
      uint64_t random_seed = Caffe::next_seed());

  /** Will not return until the internal thread has exited. */
  void StopInternalThread();
  void WaitAll();

  bool is_started(int id = 0) const {
    return threads_[id].joinable();
  }

  size_t threads_num() const {
    return threads_.size();
  }

  void resize_threads(size_t new_size) {
    threads_.resize(new_size);
  }

  void go() {
    for (shared_ptr<Flag>& flag : delay_flags_) {
      flag->set();
    }
  }

 protected:
  int target_device_;
  size_t rank_;
  void* aux_;

  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

  virtual void InternalThreadEntryN(size_t id) {}

  /* Should be tested when running loops to exit when requested. */
  bool must_stop(int id) {
    return threads_[id].interruption_requested();
  }

 private:
  void entry(int thread_id, int device, Caffe::Brew mode, uint64_t rand_seed, int solver_count,
      size_t rank, bool set_cpu_affinity);

  vector<boost::thread> threads_;
  vector<shared_ptr<Flag>> delay_flags_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
