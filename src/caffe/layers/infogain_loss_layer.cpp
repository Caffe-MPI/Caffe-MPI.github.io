#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/infogain_loss_layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void InfogainLossLayer<Ftype, Btype>::LayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  if (bottom.size() < 3) {
    CHECK(this->layer_param_.infogain_loss_param().has_source())
        << "Infogain matrix source must be specified.";
    BlobProto blob_proto;
    ReadProtoFromBinaryFile(
      this->layer_param_.infogain_loss_param().source(), &blob_proto);
    infogain_.FromProto(blob_proto);
  }
}

template <typename Ftype, typename Btype>
void InfogainLossLayer<Ftype, Btype>::Reshape(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype, Btype>::Reshape(bottom, top);
  Blob* infogain = NULL;
  if (bottom.size() < 3) {
    infogain = &infogain_;
  } else {
    infogain = bottom[2];
  }
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  CHECK_EQ(infogain->num(), 1);
  CHECK_EQ(infogain->channels(), 1);
  CHECK_EQ(infogain->height(), dim);
  CHECK_EQ(infogain->width(), dim);
}


template <typename Ftype, typename Btype>
void InfogainLossLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  const Ftype* bottom_label = bottom[1]->cpu_data<Ftype>();
  const Ftype* infogain_mat = NULL;
  if (bottom.size() < 3) {
    infogain_mat = infogain_.template cpu_data<Ftype>();
  } else {
    infogain_mat = bottom[2]->cpu_data<Ftype>();
  }
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  float loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    for (int j = 0; j < dim; ++j) {
      float prob = std::max(bottom_data[i * dim + j],
          tol<Ftype>(kLOG_THRESHOLD, min_dtype<Ftype>()));
      loss -= infogain_mat[label * dim + j] * log(prob);
    }
  }
  top[0]->mutable_cpu_data<Ftype>()[0] = loss / num;
}

template <typename Ftype, typename Btype>
void InfogainLossLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down.size() > 2 && propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to infogain inputs.";
  }
  if (propagate_down[0]) {
    const Btype* bottom_data = bottom[0]->cpu_data<Btype>();
    const Btype* bottom_label = bottom[1]->cpu_data<Btype>();
    const Btype* infogain_mat = NULL;
    if (bottom.size() < 3) {
      infogain_mat = infogain_.template cpu_data<Btype>();
    } else {
      infogain_mat = bottom[2]->cpu_data<Btype>();
    }
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    const float scale = - top[0]->cpu_diff<Btype>()[0] / num;
    for (int i = 0; i < num; ++i) {
      const int label = static_cast<int>(bottom_label[i]);
      for (int j = 0; j < dim; ++j) {
        float prob = std::max(bottom_data[i * dim + j],
            tol<Btype>(kLOG_THRESHOLD, min_dtype<Btype>()));
        bottom_diff[i * dim + j] = scale * infogain_mat[label * dim + j] / prob;
      }
    }
  }
}

INSTANTIATE_CLASS_FB(InfogainLossLayer);
REGISTER_LAYER_CLASS(InfogainLoss);
}  // namespace caffe
