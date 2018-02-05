#include <map>
#include <memory>
#include <string>

#include <google/protobuf/text_format.h>
#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/solver_factory.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class SolverFactoryTest : public MultiDeviceTest<TypeParam> {
 protected:
  SolverParameter simple_solver_param() {
    const string solver_proto =
        "train_net_param { "
        "  layer { "
        "    name: 'data' type: 'DummyData' top: 'data' "
        "    dummy_data_param { shape { dim: 1 } } "
        "  } "
        "} ";
    SolverParameter solver_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        solver_proto, &solver_param));
    return solver_param;
  }
};

TYPED_TEST_CASE(SolverFactoryTest, TestDtypesAndDevices);

TYPED_TEST(SolverFactoryTest, TestCreateSolver) {
  typename SolverRegistry::CreatorRegistry& registry = SolverRegistry::Registry();
  shared_ptr<Solver> solver;
  SolverParameter solver_param = this->simple_solver_param();
  for (typename SolverRegistry::CreatorRegistry::iterator iter =
       registry.begin(); iter != registry.end(); ++iter) {
    solver_param.set_type(iter->first);
    solver.reset(SolverRegistry::CreateSolver(solver_param));
    EXPECT_EQ(iter->first, solver->type());
  }
}

}  // namespace caffe
