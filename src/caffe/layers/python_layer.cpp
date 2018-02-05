#ifdef WITH_PYTHON_LAYER
#include "caffe/layers/python_layer.hpp"

namespace caffe {

void PyErrFatal() {
  PyErr_Print();
  std::cerr << std::endl;
  LOG(FATAL) << "Python error";
}

void PyErrReportAndForward() {
  PyObject *type_ptr = nullptr, *value_ptr = nullptr, *traceback_ptr = nullptr;
  PyErr_Fetch(&type_ptr, &value_ptr, &traceback_ptr);
  std::string err("Unknown Python error");

  bp::handle<> h_type(bp::allow_null(type_ptr));
  if (type_ptr != NULL) {
    bp::str type_pstr(h_type);
    bp::extract<std::string> e_type_pstr(type_pstr);
    if (e_type_pstr.check()) {
      err = e_type_pstr();
    } else {
      err = "Unknown exception type";
    }
  }
  bp::handle<> h_val(bp::allow_null(value_ptr));
  if (value_ptr != NULL) {
    bp::str a(h_val);
    bp::extract<std::string> returned(a);
    if (returned.check()) {
      err += returned().size() > 0 ? (": " + returned()) : "";
    } else {
      err += std::string(": Unparseable Python error: ");
    }
  }
  bp::handle<> h_tb(bp::allow_null(traceback_ptr));
  if (traceback_ptr != NULL) {
    bp::object tb(bp::import("traceback"));
    bp::object fmt_tb(tb.attr("format_tb"));
    bp::object tb_list(fmt_tb(h_tb));
    bp::object tb_str(bp::str("\n").join(tb_list));
    bp::extract<std::string> returned(tb_str);
    if (returned.check()) {
      err += ": " + returned();
    } else {
      err += std::string(": Unparseable Python traceback");
    }
  }
  LOG(ERROR) << "Python Error: " << err;
//  PyErr_Restore(type_ptr, value_ptr, traceback_ptr);
  PyErr_SetString(PyExc_RuntimeError, err.c_str());  // TODO support other types?
  bp::throw_error_already_set();
}

}
#endif
