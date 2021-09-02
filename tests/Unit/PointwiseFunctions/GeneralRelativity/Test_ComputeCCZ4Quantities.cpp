
// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <random>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CCZ4/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/CCZ4/ATilde.hpp"
#include "PointwiseFunctions/GeneralRelativity/CCZ4/ConfSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/CCZ4/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/CCZ4/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/CCZ4/PhiSquared.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace Tags {
template <typename Tag, typename Dim, typename Frame, typename>
struct deriv;
}  // namespace Tags

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
void test_compute_quantities(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-3.0, 3.0);
  const auto nn_generator = make_not_null(&generator);

  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<SpatialDim, DataType, Frame>(
          nn_generator, used_for_size);
  const auto inverse_spatial_metric_and_det = determinant_and_inverse<
      gr::Tags::DetSpatialMetric<DataType>,
      gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataType>>(
      spatial_metric);
  const Scalar<DataType>& det_spatial_metric =
      get<gr::Tags::DetSpatialMetric<DataType>>(inverse_spatial_metric_and_det);

  const auto expected_phi = pypp::call<Scalar<DataType>>(
      "GeneralRelativity.ComputeCCZ4Quantities", "phi", spatial_metric);
  CHECK_ITERABLE_APPROX((::CCZ4::phi<DataType>(det_spatial_metric)),
                        expected_phi);

  const auto expected_phi_squared = pypp::call<Scalar<DataType>>(
      "GeneralRelativity.ComputeCCZ4Quantities", "phi_squared", expected_phi);
  CHECK_ITERABLE_APPROX((::CCZ4::phi_squared<DataType>(expected_phi)),
                        expected_phi_squared);

  const auto expected_conf_spatial_metric =
      pypp::call<tnsr::ii<DataType, SpatialDim, Frame>>(
          "GeneralRelativity.ComputeCCZ4Quantities", "conformal_spatial_metric",
          expected_phi_squared, spatial_metric);
  CHECK_ITERABLE_APPROX(
      (::CCZ4::conformal_spatial_metric<SpatialDim, Frame, DataType>(
          expected_phi_squared, spatial_metric)),
      expected_conf_spatial_metric);

  // TODO: actually compute this? nah?
  const auto extrinsic_curvature =
      make_with_random_values<tnsr::ii<DataType, SpatialDim, Frame>>(
          nn_generator, make_not_null(&distribution), used_for_size);
  const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataType>>(
          inverse_spatial_metric_and_det);
  const auto expected_trace_extrinsic_curvature = pypp::call<Scalar<DataType>>(
      "GeneralRelativity.ComputeCCZ4Quantities", "trace_extrinsic_curvature",
      extrinsic_curvature, inverse_spatial_metric);
  CHECK_ITERABLE_APPROX(::CCZ4::trace_extrinsic_curvature(
                            extrinsic_curvature, inverse_spatial_metric),
                        expected_trace_extrinsic_curvature);

  const auto expected_A_tilde =
      pypp::call<tnsr::ij<DataType, SpatialDim, Frame>>(
          "GeneralRelativity.ComputeCCZ4Quantities", "a_tilde",
          expected_phi_squared, extrinsic_curvature, inverse_spatial_metric,
          spatial_metric);
  CHECK_ITERABLE_APPROX(
      ::CCZ4::a_tilde(expected_phi_squared, extrinsic_curvature,
                      expected_trace_extrinsic_curvature, spatial_metric),
      expected_A_tilde);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.CCZ4Quantities",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env("PointwiseFunctions/");

  //   GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  //   CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_quantities,
  //                                     (1, 2, 3), (Frame::Grid,
  //                                     Frame::Inertial));
  // test_compute_quantities<3,
  // Frame::Inertial>(std::numeric_limits<double>::signaling_NaN());
  test_compute_quantities<3, Frame::Inertial>(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
