#ifndef CAFFE_UTIL_BLOCKING_QUEUE_HPP_
#define CAFFE_UTIL_BLOCKING_QUEUE_HPP_

#include <queue>
#include <string>
#include <vector>
#include <functional>

namespace caffe {

template<typename T>
class BlockingQueue {
 public:
  BlockingQueue();
  ~BlockingQueue();

  void push(const T& t);
  // This logs a message if the threads needs to be blocked
  // useful for detecting e.g. when data feeding is too slow
  T pop(const char* log_on_wait);
  T pop();

  bool try_peek(T* t);
  bool try_pop(T* t);

  // Return element without removing it
  T peek();

  size_t size() const;
  bool nonblocking_size(size_t* size) const;

 protected:
  std::queue<T> queue_;
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;

  DISABLE_COPY_MOVE_AND_ASSIGN(BlockingQueue);
};

}  // namespace caffe

#endif
