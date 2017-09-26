#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template<typename Dtype>
class DataTransformer {
 public:
  DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  unsigned int Rand(int n) const {
    CHECK_GT(n, 0);
    return Rand() % n;
  }

#ifndef CPU_ONLY
  void TransformGPU(int N, int C, int H, int W, size_t sizeof_element, const Dtype* in, Dtype* out,
      const unsigned int* rands);
#endif
  void Copy(const Datum& datum, Dtype* data, size_t& out_sizeof_element);
  void Copy(const cv::Mat& datum, Dtype* data);
  void CopyPtrEntry(shared_ptr<Datum> datum, Dtype* transformed_ptr, size_t& out_sizeof_element,
      bool output_labels, Dtype* label);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, TBlob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param rand1
   *    Random value (0, RAND_MAX+1]
   * @param rand2
   *    Random value (0, RAND_MAX+1]
   * @param rand3
   *    Random value (0, RAND_MAX+1]
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void TransformPtrEntry(shared_ptr<Datum> datum, Dtype* transformed_ptr,
      std::array<unsigned int, 3> rand, bool output_labels, Dtype* label);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum>& datum_vector, TBlob<Dtype>* transformed_blob);

#ifdef USE_OPENCV
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat>& mat_vector, TBlob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, TBlob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   * @param rand1
   *    Random value (0, RAND_MAX+1]
   * @param rand2
   *    Random value (0, RAND_MAX+1]
   * @param rand3
   *    Random value (0, RAND_MAX+1]
   */
  void TransformPtr(const cv::Mat& cv_img, Dtype* transformed_ptr,
      const std::array<unsigned int, 3>& rand);
#endif  // USE_OPENCV

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum, bool use_gpu = false);

#ifdef USE_OPENCV
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img, bool use_gpu = false);
#endif  // USE_OPENCV

  void Fill3Randoms(unsigned int* rand) const;
  const TransformationParameter& transform_param() const {
    return param_;
  }

 protected:
  unsigned int Rand() const;
  void TransformGPU(const Datum& datum, Dtype* transformed_data,
      const std::array<unsigned int, 3>& rand);
  void Transform(const Datum& datum, Dtype* transformed_data,
      const std::array<unsigned int, 3>& rand);
  void TransformPtrInt(Datum& datum, Dtype* transformed_data,
      const std::array<unsigned int, 3>& rand);

  // Tranformation parameters
  TransformationParameter param_;
  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  TBlob<float> data_mean_;
  vector<float> mean_values_;
#ifndef CPU_ONLY
  GPUMemory::Workspace mean_values_gpu_;
#endif
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
