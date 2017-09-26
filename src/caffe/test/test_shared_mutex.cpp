#include <vector>
#include <thread>
#include <functional>
#include <atomic>
#include <random>
#include <memory>
#include <mutex>

#include <boost/thread.hpp>

#include "gtest/gtest.h"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template<typename Dtype>
class SharedMutexTest : public ::testing::Test {
  std::random_device mre_;
  std::uniform_int_distribution<int> dist_;
  boost::shared_mutex mutex_;

 public:
  void write_body(std::vector<int>& shelf) {
    boost::unique_lock<boost::shared_mutex> uniqueLock(mutex_);
    for (int& books : shelf) {
      books = dist_(mre_);
      books_wrote_ += books;
    }
  }

  void read_body(int reader_id, const std::vector<int>& shelf) {
    while (true) {
      if (shelf[reader_id] != 0) {  // book is now written
        shared_lock<shared_mutex> lock(mutex_);
        books_read_ += shelf[reader_id];
        break;
      }
      // giving writer some time to finish the book
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

 protected:
  SharedMutexTest() : mre_(), dist_(1, 100), books_wrote_(0), books_read_(0) {}

  virtual ~SharedMutexTest() {}

  std::atomic_int books_wrote_, books_read_;
};

TYPED_TEST_CASE(SharedMutexTest, TestDtypes);

TYPED_TEST(SharedMutexTest, TestWriteRead) {
  std::vector<int> shelf(std::thread::hardware_concurrency() - 1, 0);
  std::vector<std::thread> reader_threads(shelf.size());

  std::thread writer_thread(&SharedMutexTest<TypeParam>::write_body, this, std::ref(shelf));
  for (size_t i = 0; i < reader_threads.size(); ++i) {
    reader_threads[i] = std::thread(&SharedMutexTest<TypeParam>::read_body, this, i,
        std::cref(shelf));
  }

  for (size_t i = 0; i < reader_threads.size(); ++i) {
    reader_threads[i].join();
  }
  writer_thread.join();
  EXPECT_EQ(this->books_wrote_, this->books_read_);
}

TYPED_TEST(SharedMutexTest, TestReadWrite) {
  std::vector<int> shelf(std::thread::hardware_concurrency() - 1, 0);
  std::vector<std::thread> reader_threads(shelf.size());

  for (size_t i = 0; i < reader_threads.size(); ++i) {
    reader_threads[i] = std::thread(&SharedMutexTest<TypeParam>::read_body, this, i,
        std::cref(shelf));
  }
  std::thread writer_thread(&SharedMutexTest<TypeParam>::write_body, this, std::ref(shelf));

  for (size_t i = 0; i < reader_threads.size(); ++i) {
    reader_threads[i].join();
  }
  writer_thread.join();
  EXPECT_EQ(this->books_wrote_, this->books_read_);
}

TYPED_TEST(SharedMutexTest, TestWriteRead2) {
  std::vector<int> shelf(std::thread::hardware_concurrency() - 1, 0);
  std::vector<std::thread> reader_threads(shelf.size());

  std::thread writer_thread(&SharedMutexTest<TypeParam>::write_body, this, std::ref(shelf));
  for (size_t i = 0; i < reader_threads.size(); ++i) {
    reader_threads[i] = std::thread(&SharedMutexTest<TypeParam>::read_body, this, i,
        std::cref(shelf));
  }

  writer_thread.join();
  for (size_t i = 0; i < reader_threads.size(); ++i) {
    reader_threads[i].join();
  }
  EXPECT_EQ(this->books_wrote_, this->books_read_);
}

TYPED_TEST(SharedMutexTest, TestReadWrite2) {
  std::vector<int> shelf(std::thread::hardware_concurrency() - 1, 0);
  std::vector<std::thread> reader_threads(shelf.size());

  for (size_t i = 0; i < reader_threads.size(); ++i) {
    reader_threads[i] = std::thread(&SharedMutexTest<TypeParam>::read_body, this, i,
        std::cref(shelf));
  }
  std::thread writer_thread(&SharedMutexTest<TypeParam>::write_body, this, std::ref(shelf));

  writer_thread.join();
  for (size_t i = 0; i < reader_threads.size(); ++i) {
    reader_threads[i].join();
  }
  EXPECT_EQ(this->books_wrote_, this->books_read_);
}

}  // namespace caffe
