/**
 * @brief A solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 *
 *     SolverRegistry::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#ifndef CPU_ONLY
  #include "caffe/util/float16.hpp"
#endif

#include "caffe/solver.hpp"

namespace caffe {

class SolverRegistry {
 public:
  typedef Solver* (*Creator)(const SolverParameter&, size_t, Solver*);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry g_registry_;
    return g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    if (registry.count(type) > 0) {
      // glog is silent before calling main()
      std::cerr << "Solver type " << type << " already registered.";
    }
    registry[type] = creator;
  }

  // Get a solver using a SolverParameter.
  static Solver* CreateSolver(const SolverParameter& param, size_t rank = 0U,
      Solver* root_solver = NULL) {
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
        << " (known types: " << SolverTypeListString() << ")";
    Solver* solver = registry[type](param, rank, root_solver);
    return solver;
  }

  static vector<string> SolverTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> solver_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      solver_types.push_back(iter->first);
    }
    return solver_types;
  }

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  SolverRegistry() {}

  static string SolverTypeListString() {
    vector<string> solver_types = SolverTypeList();
    string solver_types_str;
    for (vector<string>::iterator iter = solver_types.begin();
         iter != solver_types.end(); ++iter) {
      if (iter != solver_types.begin()) {
        solver_types_str += ", ";
      }
      solver_types_str += *iter;
    }
    return solver_types_str;
  }
};


class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,
      Solver* (*creator)(const SolverParameter&, size_t, Solver*)) {
    SolverRegistry::AddCreator(type, creator);
  }
};

#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
static SolverRegisterer g_creator_f_##type(#type, creator)

#ifndef CPU_ONLY
#define REGISTER_SOLVER_CLASS(type)                                            \
  Solver* Creator_##type##Solver(                                              \
      const SolverParameter& param,                                            \
      size_t rank,                                                             \
      Solver* root_solver)                                                     \
  {                                                                            \
    const Type tp = param.solver_data_type();                                  \
    switch (tp) {                                                              \
      case FLOAT:                                                              \
        return new type##Solver<float>(param, rank, root_solver);              \
        break;                                                                 \
      case FLOAT16:                                                            \
        return new type##Solver<float16>(param, rank, root_solver);            \
        break;                                                                 \
      case DOUBLE:                                                             \
        return new type##Solver<double>(param, rank, root_solver);             \
        break;                                                                 \
      default:                                                                 \
        LOG(FATAL) << "Solver data type " << Type_Name(tp)                     \
                   << " is not supported";                                     \
    }                                                                          \
    return nullptr;                                                            \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)
#else
#define REGISTER_SOLVER_CLASS(type)                                            \
  Solver* Creator_##type##Solver(                                              \
      const SolverParameter& param,                                            \
      size_t rank,                                                             \
      Solver* root_solver)                                                     \
  {                                                                            \
    const Type tp = param.solver_data_type();                                  \
    switch (tp) {                                                              \
      case FLOAT:                                                              \
        return new type##Solver<float>(param, rank, root_solver);              \
        break;                                                                 \
      case DOUBLE:                                                             \
        return new type##Solver<double>(param, rank, root_solver);             \
        break;                                                                 \
      default:                                                                 \
        LOG(FATAL) << "Solver data type " << Type_Name(tp)                     \
                   << " is not supported";                                     \
    }                                                                          \
    return nullptr;                                                            \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)
#endif


}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_H_
