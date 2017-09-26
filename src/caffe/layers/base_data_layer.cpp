#include <map>
#include "caffe/proto/caffe.pb.h"

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
size_t BasePrefetchingDataLayer<Ftype, Btype>::threads(const LayerParameter& param) {
  // Check user's override in prototxt file
  size_t threads = param.data_param().threads();
  if (!auto_mode(param) && threads == 0U) {
    threads = 1U;  // input error fix
  }
  // 1 thread for test net
  return (auto_mode(param) || param.phase() == TEST || threads == 0U) ? 1U : threads;
}

template<typename Ftype, typename Btype>
size_t BasePrefetchingDataLayer<Ftype, Btype>::parser_threads(const LayerParameter& param) {
  // Check user's override in prototxt file
  size_t parser_threads = param.data_param().parser_threads();
  if (!auto_mode(param) && parser_threads == 0U) {
    parser_threads = 1U;  // input error fix
  }
  // 1 thread for test net
  return (auto_mode(param) || param.phase() == TEST || parser_threads == 0U) ? 1U : parser_threads;
}

template<typename Ftype, typename Btype>
bool BasePrefetchingDataLayer<Ftype, Btype>::auto_mode(const LayerParameter& param) {
  // Both should be set to positive for manual mode
  const DataParameter& dparam = param.data_param();
  bool auto_mode = !dparam.has_threads() && !dparam.has_parser_threads();
  return auto_mode;
}

template<typename Ftype, typename Btype>
BaseDataLayer<Ftype, Btype>::BaseDataLayer(const LayerParameter& param, size_t transf_num)
    : Layer<Ftype, Btype>(param), transform_param_(param.transform_param()),
      data_transformers_(transf_num) {}

template<typename Ftype, typename Btype>
void
BaseDataLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  output_labels_ = top.size() != 1;
  for (int i = 0; i < data_transformers_.size(); ++i) {
    data_transformers_[i] = make_shared<DataTransformer<Ftype>>(transform_param_, this->phase_);
    data_transformers_[i]->InitRand();
  }
  // Subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template<typename Ftype, typename Btype>
BasePrefetchingDataLayer<Ftype, Btype>::BasePrefetchingDataLayer(const LayerParameter& param)
    : BaseDataLayer<Ftype, Btype>(param, threads(param)),
      InternalThread(Caffe::current_device(), this->solver_rank_, threads(param), false),
      auto_mode_(Caffe::mode() == Caffe::GPU && auto_mode(param)),
      parsers_num_(parser_threads(param)),
      transf_num_(threads(param)),
      queues_num_(transf_num_ * parsers_num_),
      next_batch_queue_(0UL) {
  CHECK_EQ(transf_num_, threads_num());
  // We begin with minimum required
  ResizeQueues();
}

template<typename Ftype, typename Btype>
BasePrefetchingDataLayer<Ftype, Btype>::~BasePrefetchingDataLayer() {
}

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  this->rank_ = this->solver_rank_;
  bottom_init_ = bottom;
  top_init_ = top;
  BaseDataLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  const Solver* psolver = this->parent_solver();
  const uint64_t random_seed = (psolver == nullptr ||
      static_cast<uint64_t>(psolver->param().random_seed()) == Caffe::SEED_NOT_SET) ?
          Caffe::next_seed() : static_cast<uint64_t>(psolver->param().random_seed());
  StartInternalThread(false, random_seed);
}

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::InternalThreadEntry() {
  InternalThreadEntryN(0U);
}

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::InternalThreadEntryN(size_t thread_id) {
#ifndef CPU_ONLY
  const bool use_gpu_transform = this->is_gpu_transform();
#endif
  static thread_local bool iter0 = this->phase_ == TRAIN;
  if (iter0 && this->net_inititialized_flag_ != nullptr) {
    this->net_inititialized_flag_->wait();
  } else {  // nothing to wait -> initialize and start pumping
    std::lock_guard<std::mutex> lock(mutex_in_);
    InitializePrefetch();
    start_reading();
    iter0 = false;
  }
  try {
    while (!must_stop(thread_id)) {
      const size_t qid = this->queue_id(thread_id);
#ifndef CPU_ONLY
      shared_ptr<Batch<Ftype>> batch = prefetches_free_[qid]->pop();

      CHECK_EQ((size_t) -1, batch->id());
      load_batch(batch.get(), thread_id, qid);
      if (Caffe::mode() == Caffe::GPU) {
        if (!use_gpu_transform) {
          batch->data_.async_gpu_push();
        }
        if (this->output_labels_) {
          batch->label_.async_gpu_push();
        }
        CUDA_CHECK(cudaStreamSynchronize(Caffe::th_stream_aux(Caffe::STREAM_ID_ASYNC_PUSH)));
      }

      prefetches_full_[qid]->push(batch);
#else
      shared_ptr<Batch<Ftype>> batch = prefetches_free_[qid]->pop();
      load_batch(batch.get(), thread_id, qid);
      prefetches_full_[qid]->push(batch);
#endif

      if (iter0) {
        if (this->net_iteration0_flag_ != nullptr) {
          this->net_iteration0_flag_->wait();
        }
        std::lock_guard<std::mutex> lock(mutex_out_);
        if (this->net_inititialized_flag_ != nullptr) {
          this->net_inititialized_flag_ = nullptr;  // no wait on the second round
          InitializePrefetch();
          start_reading();
        }
        if (this->auto_mode_) {
          break;
        }  // manual otherwise, thus keep rolling
        iter0 = false;
      }
    }
  } catch (boost::thread_interrupted&) {
  }
}

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::ResizeQueues() {
  size_t size = prefetches_free_.size();
  if (queues_num_ > size) {
    prefetches_free_.resize(queues_num_);
    prefetches_full_.resize(queues_num_);
    for (size_t i = size; i < queues_num_; ++i) {
      shared_ptr<Batch<Ftype>> batch = make_shared<Batch<Ftype>>();
      prefetch_.push_back(batch);
      prefetches_free_[i] = make_shared<BlockingQueue<shared_ptr<Batch<Ftype>>>>();
      prefetches_full_[i] = make_shared<BlockingQueue<shared_ptr<Batch<Ftype>>>>();
      prefetches_free_[i]->push(batch);
    }
  }
  size = batch_ids_.size();
  if (transf_num_ > size) {
    batch_ids_.resize(transf_num_);
    for (size_t i = size; i < transf_num_; ++i) {
      batch_ids_[i] = i;
    }
  }
  size = this->data_transformers_.size();
  if (transf_num_ > size) {
    this->data_transformers_.resize(transf_num_);
    // some created in ctr
    for (size_t i = size; i < transf_num_; ++i) {
      this->data_transformers_[i] =
          make_shared<DataTransformer<Ftype>>(this->transform_param_, this->phase_);
      this->data_transformers_[i]->InitRand();
    }
  }
}

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::InitializePrefetch() {
  ResizeQueues();
  this->DataLayerSetUp(bottom_init_, top_init_);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  void* ptr;
  for (int i = 0; i < prefetch_.size(); ++i) {
    ptr = prefetch_[i]->data_.mutable_cpu_data();
    if (this->output_labels_) {
      ptr = prefetch_[i]->label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    const bool use_gpu_transform = this->is_gpu_transform();
    for (int i = 0; i < prefetch_.size(); ++i) {
      ptr = prefetch_[i]->data_.mutable_gpu_data();
      if (use_gpu_transform) {
        ptr = prefetch_[i]->random_vec_.mutable_gpu_data();
        ptr = prefetch_[i]->gpu_transformed_data_->template mutable_gpu_data<Ftype>();
      }
      if (this->output_labels_) {
        ptr = prefetch_[i]->label_.mutable_gpu_data();
      }
    }
  }
#else
  if (Caffe::mode() == Caffe::CPU) {
    for (int i = 0; i < prefetch_.size(); ++i) {
      ptr = prefetch_[i]->data_.mutable_cpu_data();
      if (this->output_labels_) {
        ptr = prefetch_[i]->label_.mutable_cpu_data();
      }
    }
  } else {
    NO_GPU;
  }
#endif
  (void) ptr;
  DLOG(INFO) << "[" << this->target_device_ << "] Prefetch initialized.";
}

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  // Note: this function runs in one thread per object and one object per one Solver thread
  shared_ptr<Batch<Ftype>> batch = prefetches_full_[next_batch_queue_]->pop(
      "Data layer prefetch queue empty");
  if (this->relative_iter() > 1 && top[0]->data_type() == batch->data_.data_type()
      && top[0]->shape() == batch->data_.shape()) {
    top[0]->Swap(batch->data_);
  } else {
    top[0]->CopyDataFrom(batch->data_, true);
  }
  if (this->output_labels_) {
    if (this->relative_iter() > 1 && top[1]->data_type() == batch->label_.data_type()
        && top[1]->shape() == batch->label_.shape()) {
      top[1]->Swap(batch->label_);
    } else {
      top[1]->CopyDataFrom(batch->label_, true);
    }
  }
  batch->set_id((size_t) -1);
  prefetches_free_[next_batch_queue_]->push(batch);
  next_batch_queue();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS_FB(BaseDataLayer);
INSTANTIATE_CLASS_FB(BasePrefetchingDataLayer);

}  // namespace caffe
