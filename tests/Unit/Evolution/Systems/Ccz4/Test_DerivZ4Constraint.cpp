// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/DerivZ4Constraint.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim, typename DataType>
void test_compute_grad_spatial_z4_constraint(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ij<DataType, Dim, Frame::Inertial> (*)(
          const tnsr::i<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::Ijj<DataType, Dim, Frame::Inertial>&,
          const tnsr::ijj<DataType, Dim, Frame::Inertial>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::iJ<DataType, Dim, Frame::Inertial>&)>(
          &Ccz4::grad_spatial_z4_constraint<Dim, Frame::Inertial, DataType>),
      "DerivZ4Constraint", "grad_spatial_z4_constraint", {{{-1., 1.}}},
      used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.DerivZ4Constraint",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_grad_spatial_z4_constraint,
                                    (1, 2, 3));
}
