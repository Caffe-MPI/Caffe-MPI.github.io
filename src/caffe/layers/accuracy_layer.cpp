#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void AccuracyLayer<Ftype, Btype>::LayerSetUp(
  const vector<Blob*>& bottom, const vector<Blob*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Ftype, typename Btype>
void AccuracyLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Ftype, typename Btype>
void AccuracyLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  float accuracy = 0.F;
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  const Ftype* bottom_label = bottom[1]->cpu_data<Ftype>();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  if (top.size() > 1) {
    nums_buffer_.set_data(0.F);
    top[1]->set_data(0.F);
  }
  std::vector<std::pair<float, int>> bottom_data_vector(num_labels);
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value = static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      if (top.size() > 1) {
        ++nums_buffer_.mutable_cpu_data()[label_value];
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector[k] = std::make_pair(
            static_cast<float>(bottom_data[i * dim + k * inner_num_ + j]), k);
      }

      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<float, int>>());

      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          accuracy += 1.F;
          if (top.size() > 1) {
            Ftype* top_label = top[1]->mutable_cpu_data<Ftype>();
            top_label[label_value] = top_label[label_value] + 1.;
          }
          break;
        }
      }
      ++count;
    }
  }

  top[0]->mutable_cpu_data<Ftype>()[0] = accuracy / count;
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      const float num = nums_buffer_.cpu_data()[i];
      Ftype* top_label = top[1]->mutable_cpu_data<Ftype>();
      top_label[i] = num == 0.F ? 0. : top_label[i] / num;
    }
  }
  // Accuracy layer should not be used as a loss function.
}


INSTANTIATE_CLASS_FB(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
