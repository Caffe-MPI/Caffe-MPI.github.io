#include <boost/thread.hpp>
#include <sys/sysinfo.h>

#include "caffe/util/rng.hpp"
#include "caffe/common.hpp"
#include "caffe/parallel.hpp"
#include "caffe/data_reader.hpp"

#include "caffe/clusters.hpp"

namespace caffe {

std::mutex DataReader::DataCache::cache_mutex_;
unique_ptr<DataReader::DataCache> DataReader::DataCache::data_cache_inst_;

DataReader::DataReader(const LayerParameter& param,
    size_t solver_count,
    size_t solver_rank,
    size_t parser_threads_num,
    size_t transf_threads_num,
    size_t queue_depth,
    bool sample_only,
    bool skip_one_batch,
    bool cache,
    bool shuffle)
    : InternalThread(Caffe::current_device(),
          solver_rank, sample_only ? 1U : parser_threads_num, false),
      parser_threads_num_(threads_num()),
      transf_threads_num_(sample_only ? 1U : transf_threads_num),
      queues_num_(parser_threads_num_ * transf_threads_num_),
      queue_depth_(queue_depth),
      solver_count_(solver_count),
      solver_rank_(solver_rank),
      skip_one_batch_(skip_one_batch),
      current_rec_(0),
      current_queue_(0),
      sample_only_(sample_only),
      cache_(cache && !sample_only),
      shuffle_(cache_ && shuffle) {
  CHECK(queues_num_);
  CHECK(queue_depth_);
  batch_size_ = param.data_param().batch_size();
  backend_ = param.data_param().backend();
  if (backend_ == DataParameter_DB_LEVELDB) {
    CHECK_EQ(parser_threads_num_, 1) << "LevelDB doesn't support multiple connections";
  }
  if (cache_) {
    // This is singleton, we cache TRAIN db only
    data_cache_ = DataCache::data_cache_inst(parser_threads_num_ * solver_count_, shuffle_);
  }

  free_.resize(queues_num_);
  full_.resize(queues_num_);
  LOG(INFO) << (sample_only ? "Sample " : "") << "Data Reader threads: "
      << this->threads_num() << ", out queues: " << queues_num_ << ", depth: " << queue_depth_;
  for (size_t i = 0; i < queues_num_; ++i) {
    full_[i] = make_shared<BlockingQueue<shared_ptr<Datum>>>();
    free_[i] = make_shared<BlockingQueue<shared_ptr<Datum>>>();
    for (size_t j = 0; j < queue_depth_ - 1U; ++j) {  // +1 in InternalThreadEntryN
      free_[i]->push(make_shared<Datum>());
    }
  }
  db_source_ = param.data_param().source();
  init_ = make_shared<BlockingQueue<shared_ptr<Datum>>>();
  StartInternalThread(false, Caffe::next_seed());
}

DataReader::~DataReader() {
  StopInternalThread();
}

void DataReader::InternalThreadEntry() {
  InternalThreadEntryN(0U);
}

void DataReader::InternalThreadEntryN(size_t thread_id) {
  if (cache_) {
    data_cache_->check_db(db_source_);
    data_cache_->register_new_thread();
  }
  shared_ptr<db::DB> db(db::GetDB(backend_));
  db->Open(db_source_, db::READ);
  CursorManager cm(db,
      this,
      solver_count_,
      solver_rank_,
      parser_threads_num_,
      thread_id,
      batch_size_,
      cache_ && !sample_only_,
      shuffle_ && !sample_only_);
  shared_ptr<Datum> init_datum = make_shared<Datum>();
  cm.fetch(init_datum.get());
  init_->push(init_datum);

  if (!sample_only_) {
    start_reading_flag_.wait();
  }
  cm.rewind();
  size_t skip = skip_one_batch_ ? batch_size_ : 0UL;

  size_t queue_id, ranked_rec, batch_on_solver, sample_count = 0UL;
  shared_ptr<Datum> datum = make_shared<Datum>();
  try {
    while (!must_stop(thread_id)) {
      cm.next(datum);
      // See comment below
      ranked_rec = (size_t) datum->record_id() / cm.full_cycle();
      batch_on_solver = ranked_rec * parser_threads_num_ + thread_id;
      queue_id = batch_on_solver % queues_num_;

      if (thread_id == 0 && skip > 0U) {
        --skip;
        continue;
      }

      full_push(queue_id, datum);

      if (sample_only_) {
        ++sample_count;
        if (sample_count >= batch_size_) {
          // sample batch complete
          break;
        }
      }
      datum = free_pop(queue_id);
    }
  } catch (boost::thread_interrupted&) {
  }
}

shared_ptr<Datum>& DataReader::DataCache::next_new() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_buffer_.emplace_back(make_shared<Datum>());
  return cache_buffer_.back();
}

shared_ptr<Datum>& DataReader::DataCache::next_cached() {
  if (just_cached_.load()) {
    cache_bar_.wait();
    just_cached_.store(false);
    LOG_FIRST_N(INFO, 1) << "Cached " << cache_buffer_.size() << " records by "
          << cached_flags_.size() << " threads";
#ifdef DEBUG
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      std::multiset<size_t> pk;
      for (auto &entry : cache_buffer_) {
        pk.insert(entry->record_id());
        if (pk.count(entry->record_id()) > 1) {
          LOG(ERROR) << "Record " << entry->record_id() << " duplicated "
              << entry->record_id() << " times";
        }
      }
      LOG(INFO) << "Recorded " << pk.size() << " from " << *pk.begin() << " to " << *pk.rbegin();
    }
#endif
    cache_bar_.wait();
  }
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (shuffle_ && cache_idx_== 0UL) {
    LOG(INFO) << "Shuffling " << cache_buffer_.size() << " records...";
    caffe::shuffle(cache_buffer_.begin(), cache_buffer_.end());
  }
  shared_ptr<Datum>& datum = cache_buffer_[cache_idx_++];
  if (cache_idx_ >= cache_buffer_.size()) {
    cache_idx_= 0UL;
  }
  return datum;
}

void DataReader::DataCache::just_cached() {
  just_cached_.store(true);
  cached_flags_[std::this_thread::get_id()]->set();
}

bool DataReader::DataCache::check_memory() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  bool mem_ok = true;
  if (cache_buffer_.size() > 0UL && cache_buffer_.size() % 10000UL == 0UL) {
    struct sysinfo sinfo;
    sysinfo(&sinfo);
    if (sinfo.totalswap > 0UL && sinfo.freeswap < sinfo.totalswap / 2UL) {
      LOG(WARNING) << "Data Reader cached " << cache_buffer_.size()
                   << " records so far but it can't continue because it used more than half"
                   << " of swap buffer. Free swap memory left: " << sinfo.freeswap << " of total "
                   << sinfo.totalswap << ". Cache and shuffling are now disabled.";
      mem_ok = false;
    }
    if (sinfo.totalswap == 0UL && sinfo.freeram < sinfo.totalram / 1000UL) {
      LOG(WARNING) << "Data Reader cached " << cache_buffer_.size()
                   << " records so far but it can't continue because it used more than 99.9%"
                   << " of RAM and there is no swap space available. Free RAM left: "
                   << sinfo.freeram << " of total " << sinfo.totalram
                   << ". Cache and shuffling are now disabled.";
      mem_ok = false;
    }
    if (!mem_ok) {
      cache_buffer_.clear();
      shuffle_ = false;
    }
  }
  return mem_ok;
}

DataReader::CursorManager::CursorManager(shared_ptr<db::DB> db, DataReader* reader,
    size_t solver_count, size_t solver_rank, size_t parser_threads, size_t parser_thread_id,
    size_t batch_size, bool cache, bool shuffle)
    : db_(db),
      cursor_(db->NewCursor()),
      reader_(reader),
      solver_count_(solver_count),
      solver_rank_(solver_rank),
      batch_size_(batch_size),
      parser_threads_(parser_threads),
      parser_thread_id_(parser_thread_id),
      rank_cycle_(parser_threads_ * batch_size_),
      full_cycle_(rank_cycle_ * solver_count_ * Clusters::node_count()),
      rec_id_(0UL),
      rec_end_(0UL),
      cache_(cache),
      shuffle_(shuffle),
      cached_all_(false) {}

DataReader::CursorManager::~CursorManager() {
  cursor_.reset();
  db_->Close();
}

void DataReader::CursorManager::next(shared_ptr<Datum>& datum) {
  if (cached_all_) {
    datum = reader_->next_cached();
  } else {
    while (cache_) {
      if (!reader_->check_memory()) {
        cache_ = false;
        shuffle_ = false;
        break;
      }
      datum = reader_->next_new();
      break;
    }
    fetch(datum.get());
  }

  datum->set_record_id(rec_id_);
  size_t old_id = rec_id_;
  ++rec_id_;
  if (rec_id_ == rec_end_) {
    rec_id_ += full_cycle_ - batch_size_;
    rec_end_ += full_cycle_;
  }
  if (cached_all_) {
    return;
  }
  for (size_t i = old_id; i < rec_id_; ++i) {
    cursor_->Next();
    if (!cursor_->valid()) {
      if (cache_) {
        cached_all_ = true;
        reader_->just_cached();
        break;  // we cache first epoch, then we just read it from cache
      }
      LOG_IF(INFO, solver_rank_ == 0 && parser_thread_id_ == 0) << "Restarting data pre-fetching";
      cursor_->SeekToFirst();
    }
  }
}

/*
  Example: 2 solvers (rank 0, rank 1), 3 parser threads per solver (pt0, pt1, pt2),
           2 transformer threads per solver (tr0, tr1) - each transformer owns queue set
           with number of queues equal to the number of parser threads)

        B0    B1    B2    B3    B4    B5    B6    B7
      --------------------------------------------------  --> S0.tr0 S0.q0
   |                                                      --> S0.tr0 S0.q1
   |  r0pt0.q0                            r0pt0.q3        --> S0.tr0 S0.q2
S0 |        r0pt1.q1                            r0pt1.q4  --> S0.tr1 S0.q3
   |              r0pt2.q2                                --> S0.tr1 S0.q4
   |                                                      --> S0.tr1 S0.q5
      ..................................................
   |                                                      --> S1.tr0 S1.q0
   |                    r1pt0.q0                          --> S1.tr0 S1.q1
S1 |                          r1pt1.q1                    --> S1.tr0 S1.q2
   |                                r1pt2.q2              --> S1.tr1 S1.q3
   |                                                      --> S1.tr1 S1.q4
      --------------------------------------------------  --> S1.tr1 S1.q5
      <-- rank cycle ->
      <---------- full cycle ----------->
*/
void DataReader::CursorManager::rewind() {
  CHECK(parser_threads_);

  size_t rank_cycle_per_solver = parser_threads_ * batch_size_;
  size_t rank_cycle_per_node = rank_cycle_per_solver * solver_count_;
  size_t rank_cycle_begin = rank_cycle_per_solver * solver_rank_
                        + rank_cycle_per_node * Clusters::node_rank();

  //size_t rank_cycle_begin = rank_cycle_ * solver_rank_;
  rec_id_ = rank_cycle_begin + parser_thread_id_ * batch_size_;
  rec_end_ = rec_id_ + batch_size_;
  cursor_->SeekToFirst();
  for (size_t i = 0; i < rec_id_; ++i) {
    cursor_->Next();
    if (!cursor_->valid()) {
      cursor_->SeekToFirst();
    }
  }
}

void DataReader::CursorManager::fetch(Datum* datum) {
  if (!datum->ParseFromArray(cursor_->data(), (int) cursor_->size())) {
    LOG(ERROR) << "Database cursor failed to parse Datum record";
  }
}

}  // namespace caffe
