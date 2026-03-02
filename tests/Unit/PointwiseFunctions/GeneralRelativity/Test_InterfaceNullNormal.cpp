// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/GeneralRelativity/InterfaceNullNormal.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <size_t SpatialDim, typename DataType>
tnsr::a<DataType, SpatialDim, Frame::Inertial>
interface_outgoing_null_normal_one_form(
    const tnsr::a<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_one_form,
    const tnsr::i<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_one_form,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>& shift) {
  return gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
      spacetime_normal_one_form, interface_normal_one_form, shift, 1.);
}
template <size_t SpatialDim, typename DataType>
tnsr::a<DataType, SpatialDim, Frame::Inertial>
interface_incoming_null_normal_one_form(
    const tnsr::a<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_one_form,
    const tnsr::i<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_one_form,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>& shift) {
  return gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
      spacetime_normal_one_form, interface_normal_one_form, shift, -1.);
}
template <size_t SpatialDim, typename DataType>
tnsr::A<DataType, SpatialDim, Frame::Inertial>
interface_outgoing_null_normal_vector(
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_vector) {
  return gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
      spacetime_normal_vector, interface_normal_vector, 1.);
}
template <size_t SpatialDim, typename DataType>
tnsr::A<DataType, SpatialDim, Frame::Inertial>
interface_incoming_null_normal_vector(
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_vector) {
  return gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
      spacetime_normal_vector, interface_normal_vector, -1.);
}

template <size_t SpatialDim, typename DataType>
void test_interface_null_normals(const DataType& used_for_size) {
  {
    auto* f = &interface_outgoing_null_normal_one_form<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_outgoing_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
  {
    auto* f = &interface_outgoing_null_normal_vector<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_outgoing_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
  {
    auto* f = &interface_incoming_null_normal_one_form<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_incoming_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
  {
    auto* f = &interface_incoming_null_normal_vector<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_incoming_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
}

template <typename DataType>
void test_one_form_shift_time_component(const DataType& used_for_size) {
  constexpr size_t spatial_dim = 3;
  const auto alpha = make_with_value<DataType>(used_for_size, 1.7);

  tnsr::a<DataType, spatial_dim, Frame::Inertial> spacetime_normal_one_form{
      get_size(used_for_size)};
  spacetime_normal_one_form.get(0) = -alpha;
  for (size_t i = 0; i < spatial_dim; ++i) {
    spacetime_normal_one_form.get(i + 1) =
        make_with_value<DataType>(used_for_size, 0.0);
  }

  std::array<double, spatial_dim> shift_values{{0.2, -0.3, 0.4}};
  tnsr::i<DataType, spatial_dim, Frame::Inertial> interface_normal_one_form{
      get_size(used_for_size)};
  tnsr::I<DataType, spatial_dim, Frame::Inertial> shift{
      get_size(used_for_size)};
  for (size_t i = 0; i < spatial_dim; ++i) {
    interface_normal_one_form.get(i) =
        make_with_value<DataType>(
            used_for_size, 0.5 + 0.1 * static_cast<double>(i));
    shift.get(i) =
        make_with_value<DataType>(used_for_size, gsl::at(shift_values, i));
  }

  auto incoming =
      gr::interface_null_normal<DataType, spatial_dim, Frame::Inertial>(
          spacetime_normal_one_form, interface_normal_one_form, shift, -1.0);
  auto outgoing =
      gr::interface_null_normal<DataType, spatial_dim, Frame::Inertial>(
          spacetime_normal_one_form, interface_normal_one_form, shift, 1.0);

  DataType shift_dot_interface = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < spatial_dim; ++i) {
    shift_dot_interface +=
        interface_normal_one_form.get(i) *
        make_with_value<DataType>(used_for_size, gsl::at(shift_values, i));
  }
  const auto expected_time_diff = sqrt(2.0) * shift_dot_interface;
  const auto actual_time_diff = outgoing.get(0) - incoming.get(0);
  CHECK(max(abs(actual_time_diff - expected_time_diff)) < 1.0e-12);

  DataType orthogonality_contraction =
      (outgoing.get(0) - incoming.get(0)) / alpha;
  for (size_t i = 0; i < spatial_dim; ++i) {
    orthogonality_contraction +=
        make_with_value<DataType>(used_for_size, -gsl::at(shift_values, i)) /
        alpha * (outgoing.get(i + 1) - incoming.get(i + 1));
  }
  CHECK(max(abs(orthogonality_contraction)) < 1.0e-12);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.IntfcNullNormals",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_interface_null_normals, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_one_form_shift_time_component, ());
}
