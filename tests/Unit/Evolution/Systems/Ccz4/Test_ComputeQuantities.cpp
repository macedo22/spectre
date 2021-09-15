// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/DerivLapse.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim, typename DataType>
void test_compute_grad_grad_lapse(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ii<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&,
          const tnsr::Ijj<DataType, Dim, Frame::Inertial>&,
          const tnsr::i<DataType, Dim, Frame::Inertial>&,
          const tnsr::ij<DataType, Dim, Frame::Inertial>&)>(
          &::Ccz4::grad_grad_lapse<Dim, Frame::Inertial, DataType>),
      "ComputeQuantities", "grad_grad_lapse", {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.ComputeQuantities",
                  "[Evolution][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_grad_grad_lapse, (1, 2, 3));

  // Check that compute items work correctly in the DataBox
  // First, check that the names are correct
  TestHelpers::db::test_compute_tag<
      Ccz4::Tags::GradGradLapseCompute<3, Frame::Inertial, DataVector>>(
      "GradGradLapse");

  // Next, check that the compute items return the correct values
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-0.1, 0.1);

  const size_t num_pts = 5;
  const DataVector used_for_size(num_pts);

  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const auto christoffel_second_kind =
      make_with_random_values<tnsr::Ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const auto field_a =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const auto d_field_a =
      make_with_random_values<tnsr::ij<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  const auto expected_grad_grad_lapse =
      Ccz4::grad_grad_lapse(lapse, christoffel_second_kind, field_a, d_field_a);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::Lapse<DataVector>,
          gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial,
                                                 DataVector>,
          Ccz4::Tags::FieldA<3, Frame::Inertial, DataVector>,
          ::Tags::deriv<Ccz4::Tags::FieldA<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>,
      db::AddComputeTags<
          Ccz4::Tags::GradGradLapseCompute<3, Frame::Inertial, DataVector>>>(
      lapse, christoffel_second_kind, field_a, d_field_a);
  CHECK(db::get<Ccz4::Tags::GradGradLapse<3, Frame::Inertial, DataVector>>(
            box) == expected_grad_grad_lapse);
}
