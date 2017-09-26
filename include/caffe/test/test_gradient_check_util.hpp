#ifndef CAFFE_TEST_GRADIENT_CHECK_UTIL_H_
#define CAFFE_TEST_GRADIENT_CHECK_UTIL_H_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

// The gradient checker adds a L2 normalization loss function on top of the
// top blobs, and checks the gradient.
template<typename Dtype>
class GradientChecker {
 public:
  // kink and kink_range specify an ignored nonsmooth region of the form
  // kink - kink_range <= |feature value| <= kink + kink_range,
  // which accounts for all nonsmoothness in use by caffe
  GradientChecker(const float stepsize, const float threshold, const unsigned int seed = 1701,
      const float kink = 0., const float kink_range = -1) : stepsize_(stepsize),
                                                            threshold_(threshold), kink_(kink),
                                                            kink_range_(kink_range), seed_(seed) {}

  // Checks the gradient of a layer, with provided bottom layers and top
  // layers.
  // Note that after the gradient check, we do not guarantee that the data
  // stored in the layer parameters and the blobs are unchanged.
  void CheckGradient(LayerBase* layer, const vector<Blob*>& bottom, const vector<Blob*>& top,
      int check_bottom = -1) {
    layer->SetUp(bottom, top);
    CheckGradientSingle(layer, bottom, top, check_bottom, -1, -1);
  }

  void
  CheckGradientExhaustive(LayerBase* layer, const vector<Blob*>& bottom, const vector<Blob*>& top,
      int check_bottom = -1);

  // CheckGradientEltwise can be used to test layers that perform element-wise
  // computation only (e.g., neuron layers) -- where (d y_i) / (d x_j) = 0 when
  // i != j.
  void
  CheckGradientEltwise(LayerBase* layer, const vector<Blob*>& bottom, const vector<Blob*>& top);

  // Checks the gradient of a single output with respect to particular input
  // blob(s).  If check_bottom = i >= 0, check only the ith bottom TBlob.
  // If check_bottom == -1, check everything -- all bottom Blobs and all
  // param Blobs.  Otherwise (if check_bottom < -1), check only param Blobs.
  void CheckGradientSingle(LayerBase* layer, const vector<Blob*>& bottom, const vector<Blob*>& top,
      int check_bottom, int top_id, int top_data_id, bool element_wise = false);

  // Checks the gradient of a network. This network should not have any data
  // layers or loss layers, since the function does not explicitly deal with
  // such cases yet. All input blobs and parameter blobs are going to be
  // checked, layer-by-layer to avoid numerical problems to accumulate.
  void CheckGradientNet(Net& net, const vector<Blob*>& input);

 protected:
  float GetObjAndGradient(const LayerBase& layer, const vector<Blob*>& top, int top_id = -1,
      int top_data_id = -1);
  float stepsize_, threshold_, kink_, kink_range_;
  unsigned int seed_;
};


template<typename Dtype>
void GradientChecker<Dtype>::CheckGradientSingle(LayerBase* layer, const vector<Blob*>& bottom,
    const vector<Blob*>& top, int check_bottom, int top_id, int top_data_id, bool element_wise) {
  if (element_wise) {
    CHECK_EQ(0, layer->blobs().size());
    CHECK_LE(0, top_id);
    CHECK_LE(0, top_data_id);
    const int top_count = top[top_id]->count();
    for (int blob_id = 0; blob_id < bottom.size(); ++blob_id) {
      CHECK_EQ(top_count, bottom[blob_id]->count());
    }
  }

  // First, figure out what blobs we need to check against, and zero init
  // parameter blobs.
  vector<Blob*> blobs_to_check;
  vector<bool> propagate_down(bottom.size(), check_bottom == -1);
  for (int i = 0; i < layer->blobs().size(); ++i) {
    Blob* blob = layer->blobs()[i].get();
    blob->set_diff(0.F);
    blobs_to_check.push_back(blob);
  }
  if (check_bottom == -1) {
    for (int i = 0; i < bottom.size(); ++i) {
      blobs_to_check.push_back(bottom[i]);
    }
  } else if (check_bottom >= 0) {
    CHECK_LT(check_bottom, bottom.size());
    blobs_to_check.push_back(bottom[check_bottom]);
    propagate_down[check_bottom] = true;
  }
  CHECK_GT(blobs_to_check.size(), 0) << "No blobs to check.";
  // Compute the gradient analytically using Backward
  Caffe::set_random_seed(seed_);
  // Ignore the loss from the layer (it's just the weighted sum of the losses
  // from the top blobs, whose gradients we may want to test individually).
  layer->Forward(bottom, top);
  // Get additional loss from the objective
  GetObjAndGradient(*layer, top, top_id, top_data_id);

  layer->Backward(top, propagate_down, bottom);

  // Store computed gradients for all checked blobs
  vector<shared_ptr<TBlob<Dtype>>> computed_gradient_blobs(blobs_to_check.size());
  for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
    Blob* current_blob = blobs_to_check[blob_id];
    computed_gradient_blobs[blob_id].reset(new TBlob<Dtype>());
    computed_gradient_blobs[blob_id]->ReshapeLike(*current_blob);
    const int count = blobs_to_check[blob_id]->count();
    const Dtype* diff = blobs_to_check[blob_id]->cpu_diff<Dtype>();
    Dtype* computed_gradients = computed_gradient_blobs[blob_id]->mutable_cpu_data();
    caffe_copy(count, diff, computed_gradients);
  }
  // Compute derivative of top w.r.t. each bottom and parameter input using
  // finite differencing.
  for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
    Blob* current_blob = blobs_to_check[blob_id];
    const Dtype* computed_gradients = computed_gradient_blobs[blob_id]->cpu_data();
    for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) {
      float feature_val = current_blob->cpu_data<float>()[feat_id];
      // For an element-wise layer, we only need to do finite differencing to
      // compute the derivative of top[top_id][top_data_id] w.r.t.
      // bottom[blob_id][i] only for i == top_data_id.  For any other
      // i != top_data_id, we know the derivative is 0 by definition, and simply
      // check that that's true.
      float estimated_gradient = 0;
      float positive_objective = 0;
      float negative_objective = 0;
      if (!element_wise || (feat_id == top_data_id)) {
        // Do finite differencing.
        // Compute loss with stepsize_ added to input.
        feature_val += stepsize_;
        current_blob->mutable_cpu_data<float>()[feat_id] = feature_val;
        Caffe::set_random_seed(seed_);
        layer->Forward(bottom, top);
        positive_objective = GetObjAndGradient(*layer, top, top_id, top_data_id);
        // Compute loss with stepsize_ subtracted from input.
        feature_val -= stepsize_ * 2.F;
        current_blob->mutable_cpu_data<float>()[feat_id] = feature_val;
        Caffe::set_random_seed(seed_);
        layer->Forward(bottom, top);
        negative_objective = GetObjAndGradient(*layer, top, top_id, top_data_id);
        // Recover original input value.
        feature_val += stepsize_;
        current_blob->mutable_cpu_data<float>()[feat_id] = feature_val;
        estimated_gradient = (positive_objective - negative_objective) / stepsize_ / 2.;
      }
      float computed_gradient = computed_gradients[feat_id];
      float feature = current_blob->cpu_data<float>()[feat_id];
      if (kink_ - kink_range_ > fabs(feature) || fabs(feature) > kink_ + kink_range_) {
        // We check relative accuracy, but for too small values, we threshold
        // the scale factor by 1.
        float scale = std::max<float>(std::max(fabs(computed_gradient), fabs(estimated_gradient)),
            float(1.));
        EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
                  << "debug: (top_id, top_data_id, blob_id, feat_id)=" << top_id << ","
                  << top_data_id << "," << blob_id << ", " << feat_id << "; feat = " << feature
                  << "; objective+ = " << positive_objective << "; objective- = "
                  << negative_objective;
      }
    }
  }
}

template<typename Dtype>
void GradientChecker<Dtype>::CheckGradientExhaustive(LayerBase* layer, const vector<Blob*>& bottom,
    const vector<Blob*>& top, int check_bottom) {
  layer->SetUp(bottom, top);
  CHECK_GT(top.size(), 0) << "Exhaustive mode requires at least one top blob.";
  for (int i = 0; i < top.size(); ++i) {
    for (int j = 0; j < top[i]->count(); ++j) {
      CheckGradientSingle(layer, bottom, top, check_bottom, i, j);
    }
  }
}

template<typename Dtype>
void GradientChecker<Dtype>::CheckGradientEltwise(LayerBase* layer, const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  layer->SetUp(bottom, top);
  CHECK_GT(top.size(), 0) << "Eltwise mode requires at least one top blob.";
  const int check_bottom = -1;
  const bool element_wise = true;
  for (int i = 0; i < top.size(); ++i) {
    for (int j = 0; j < top[i]->count(); ++j) {
      CheckGradientSingle(layer, bottom, top, check_bottom, i, j, element_wise);
    }
  }
}

template<typename Dtype>
void GradientChecker<Dtype>::CheckGradientNet(Net& net, const vector<Blob*>& input) {
  const vector<shared_ptr<LayerBase>>& layers = net.layers();
  vector<vector<Blob*>>& bottom_vecs = net.bottom_vecs();
  vector<vector<Blob*>>& top_vecs = net.top_vecs();
  for (int i = 0; i < layers.size(); ++i) {
    net.Forward(input);
    LOG(ERROR) << "Checking gradient for " << layers[i]->layer_param().name();
    CheckGradientExhaustive(layers[i].get(), bottom_vecs[i], top_vecs[i]);
  }
}

template<typename Dtype>
float GradientChecker<Dtype>::GetObjAndGradient(const LayerBase& layer, const vector<Blob*>& top,
    int top_id, int top_data_id) {
  float loss = 0.F;
  if (top_id < 0) {
    // the loss will be half of the sum of squares of all outputs
    for (int i = 0; i < top.size(); ++i) {
      Blob* top_blob = top[i];
      const float* top_blob_data = top_blob->cpu_data<float>();
      float* top_blob_diff = top_blob->mutable_cpu_diff<float>();
      int count = top_blob->count();
      for (int j = 0; j < count; ++j) {
        loss += top_blob_data[j] * top_blob_data[j];
      }
      // set the diff: simply the data.
      caffe_copy(top_blob->count(), top_blob_data, top_blob_diff);
    }
    loss /= 2.F;
  } else {
    // the loss will be the top_data_id-th element in the top_id-th blob.
    for (int i = 0; i < top.size(); ++i) {
      top[i]->set_diff(0.F);
    }
    const float loss_weight = 2.F;
    loss = top[top_id]->cpu_data<float>()[top_data_id] * loss_weight;
    top[top_id]->mutable_cpu_diff<float>()[top_data_id] = loss_weight;
  }
  return loss;
}

}  // namespace caffe

#endif  // CAFFE_TEST_GRADIENT_CHECK_UTIL_H_
