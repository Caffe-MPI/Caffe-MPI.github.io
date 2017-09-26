#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

class PyGILAquire {
  PyGILState_STATE state_;
 public:
  PyGILAquire() {
    state_ = PyGILState_Ensure();
  }
  ~PyGILAquire() {
    PyGILState_Release(state_);
  }
DISABLE_COPY_MOVE_AND_ASSIGN(PyGILAquire);
};

class PyGILRelease {
  PyThreadState *state_;
 public:
  PyGILRelease() {
    state_ = PyEval_SaveThread();
  }
  ~PyGILRelease() {
    PyEval_RestoreThread(state_);
  }
DISABLE_COPY_MOVE_AND_ASSIGN(PyGILRelease);
};

namespace caffe {

void PyErrFatal();
void PyErrReportAndForward();


#define PYTHON_CALL_BEGIN  \
{                          \
  PyGILRelease pygr;       \
  {                        \
    PyGILAquire pgil;

#define PYTHON_CALL_END    \
  }                        \
}


template <typename Ftype, typename Btype>
class PythonLayer : public Layer<Ftype, Btype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Ftype, Btype>(param), self_(bp::handle<>(bp::borrowed(self))) {}

  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override {
    try {
      std::lock_guard<std::mutex> lock(mutex());
      PYTHON_CALL_BEGIN
      self_.attr("param_str") = bp::str(this->layer_param_.python_param().param_str());
      self_.attr("phase") = static_cast<int>(this->phase_);
      self_.attr("setup")(bottom, top);
      PYTHON_CALL_END
    } catch (const bp::error_already_set&) {
      PyErrReportAndForward();
    } catch (...) {
      PyErrFatal();
    }
  }

  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override {
    try {
      std::lock_guard<std::mutex> lock(mutex());
      PYTHON_CALL_BEGIN
      self_.attr("reshape")(bottom, top);
      PYTHON_CALL_END
    } catch (const bp::error_already_set&) {
      PyErrReportAndForward();
    } catch (...) {
      PyErrFatal();
    }
  }

  inline bool ShareInParallel() const override {
    return this->layer_param_.python_param().share_in_parallel();
  }

  inline const char* type() const override { return "Python"; }

  static std::mutex& mutex() {
    return mutex_;
  }

 protected:
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override {
    try {
      std::lock_guard<std::mutex> lock(mutex());
      PYTHON_CALL_BEGIN
      self_.attr("forward")(bottom, top);
      PYTHON_CALL_END
    } catch (const bp::error_already_set&) {
      PyErrReportAndForward();
    } catch (...) {
      PyErrFatal();
    }
  }

  void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) override {
    try {
      std::lock_guard<std::mutex> lock(mutex());
      PYTHON_CALL_BEGIN
      self_.attr("backward")(top, propagate_down, bottom);
      PYTHON_CALL_END
    } catch (const bp::error_already_set&) {
      PyErrReportAndForward();
    } catch (...) {
      PyErrFatal();
    }
  }

 private:
  bp::object self_;
  static std::mutex mutex_;
};

template <typename Ftype, typename Btype> std::mutex PythonLayer<Ftype, Btype>::mutex_;

}  // namespace caffe

#endif
