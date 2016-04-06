#include "caffe/bloblite.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Bloblite<Dtype>::Reshape(const int num) {
  CHECK_GE(num, 0);
  num_ = num;
  count_ = num_; 
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void Bloblite<Dtype>::ReshapeLike(const Bloblite<Dtype>& other) {
  Reshape(other.num());
}

template <typename Dtype>
Bloblite<Dtype>::Bloblite(const int num)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num);
}

template <typename Dtype>
const Dtype* Bloblite<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Bloblite<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Bloblite<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
Dtype* Bloblite<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Bloblite<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
void Bloblite<Dtype>::ShareData(const Bloblite& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <> unsigned int Bloblite<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Bloblite<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Bloblite<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}


INSTANTIATE_CLASS(Bloblite);
template class Bloblite<int>;
template class Bloblite<unsigned int>;

}  // namespace caffe

