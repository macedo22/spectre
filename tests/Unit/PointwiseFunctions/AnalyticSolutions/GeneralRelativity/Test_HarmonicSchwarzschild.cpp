// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace {
// Get test coordinates
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.32;
  get<1>(x) = 0.82;
  get<2>(x) = 1.24;
  return x;
}

template <typename Frame, typename DataType>
void test_tag_retrieval(const DataType& used_for_size) {
  // Parameters for HarmonicSchwarzschild solution
  const double mass = 1.234;
  const std::array<double, 3> center{{1.0, 2.0, 3.0}};
  const auto x = spatial_coords<Frame>(used_for_size);
  const double t = 1.3;

  // Evaluate solution
  const gr::Solutions::HarmonicSchwarzschild solution(mass, center);
  TestHelpers::AnalyticSolutions::test_tag_retrieval(
      solution, x, t,
      typename gr::Solutions::HarmonicSchwarzschild::template tags<DataType,
                                                                   Frame>{});
}

void test_serialize() {
  gr::Solutions::HarmonicSchwarzschild solution(3.0, {{0.0, 3.0, 4.0}});
  test_serialization(solution);
}

void test_copy_and_move() {
  gr::Solutions::HarmonicSchwarzschild solution(3.0, {{0.0, 3.0, 4.0}});
  test_copy_semantics(solution);
  auto solution_copy = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}

void test_construct_from_options() {
  const auto created =
      TestHelpers::test_creation<gr::Solutions::HarmonicSchwarzschild>(
          "Mass: 0.5\n"
          "Center: [1.0,3.0,2.0]");
  CHECK(created ==
        gr::Solutions::HarmonicSchwarzschild(0.5, {{1.0, 3.0, 2.0}}));
}

// Test that computed spacetime quantities are computed as expected. See
// documentation for `gr::Solutions::HarmonicSchwarzschild` to see equations for
// expected quantities.
template <typename Frame, typename DataType>
void test_computed_quantities(const DataType used_for_size) {
  // Parameters for HarmonicSchwarzschild solution
  const double mass = 1.03;
  const std::array<double, 3> center{{0.2, -0.1, 0.4}};
  const auto x = spatial_coords<Frame>(used_for_size);
  const double t = 1.3;

  // Evaluate solution
  gr::Solutions::HarmonicSchwarzschild solution(mass, center);

  // Get solution's spacetime quantities
  const auto vars = solution.variables(
      x, t,
      typename gr::Solutions::HarmonicSchwarzschild::tags<DataType, Frame>{});
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  const auto& d_lapse =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivLapse<DataType,
                                                                    Frame>>(
          vars);
  const auto& shift = get<gr::Tags::Shift<3, Frame, DataType>>(vars);
  const auto& d_shift =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivShift<DataType,
                                                                    Frame>>(
          vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<3, Frame, DataType>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame, DataType>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>>>(vars);
  const auto& d_spatial_metric =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivSpatialMetric<
          DataType, Frame>>(vars);
  const auto& sqrt_det_spatial_metric =
      get<typename gr::Tags::SqrtDetSpatialMetric<DataType>>(vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<3, Frame, DataType>>(vars);
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<3, Frame, DataType>>(vars);

  // Check those quantities that should be zero
  const auto zero = make_with_value<DataType>(x, 0.);
  CHECK(dt_lapse.get() == zero);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(dt_shift.get(i) == zero);
    for (size_t j = 0; j < 3; ++j) {
      CHECK(dt_spatial_metric.get(i, j) == zero);
    }
  }

  // Check remaining quantities

  tnsr::I<DataType, 3, Frame> expected_x_minus_center{};
  for (size_t i = 0; i < 3; ++i) {
    expected_x_minus_center.get(i) = x.get(i) - gsl::at(center, i);
  }

  const DataType expected_r = get(magnitude(expected_x_minus_center));
  const DataType expected_one_over_r_squared = 1.0 / square(expected_r);
  const DataType expected_one_over_r_cubed = 1.0 / cube(expected_r);
  const DataType expected_two_m_over_m_plus_r =
      2.0 * mass / (mass + expected_r);
  const DataType expected_spatial_metric_rr =
      1.0 + expected_two_m_over_m_plus_r +
      square(expected_two_m_over_m_plus_r) + cube(expected_two_m_over_m_plus_r);
  const DataType expected_d_spatial_metric_rr =
      -1.0 / (2.0 * mass) * square(expected_two_m_over_m_plus_r) -
      (1.0 / mass) * cube(expected_two_m_over_m_plus_r) -
      (3.0 / (2.0 * mass)) * pow<4>(expected_two_m_over_m_plus_r);
  const DataType expected_f_0 = square(1 + mass / expected_r);
  const DataType expected_d_f_0 =
      2.0 * (1 + mass / expected_r) * (-mass * expected_one_over_r_squared);
  const DataType expected_f_1 =
      (expected_spatial_metric_rr - expected_f_0) / expected_r;
  const DataType expected_f_2 =
      expected_d_spatial_metric_rr - expected_d_f_0 - 2.0 * expected_f_1;
  const DataType expected_f_3 = square(expected_two_m_over_m_plus_r) /
                                (expected_r * expected_spatial_metric_rr);
  const DataType expected_f_4 =
      -expected_f_3 -
      (1.0 / mass) * cube(expected_two_m_over_m_plus_r) /
          expected_spatial_metric_rr -
      expected_d_spatial_metric_rr *
          square((expected_two_m_over_m_plus_r) / expected_spatial_metric_rr);

  auto expected_lapse = make_with_value<Scalar<DataType>>(x, 0.0);
  get(expected_lapse) = 1.0 / sqrt(expected_spatial_metric_rr);
  CHECK_ITERABLE_APPROX(lapse, expected_lapse);

  tnsr::i<DataType, 3, Frame> expected_d_lapse{};
  for (size_t i = 0; i < 3; ++i) {
    expected_d_lapse.get(i) = -0.5 * cube(get(expected_lapse)) *
                              expected_d_spatial_metric_rr *
                              expected_x_minus_center.get(i) / expected_r;
  }
  CHECK_ITERABLE_APPROX(d_lapse, expected_d_lapse);

  tnsr::I<DataType, 3, Frame> expected_shift{};
  for (size_t i = 0; i < 3; ++i) {
    expected_shift.get(i) = expected_two_m_over_m_plus_r *
                            expected_x_minus_center.get(i) /
                            (expected_r * expected_spatial_metric_rr);
  }
  CHECK_ITERABLE_APPROX(shift, expected_shift);

  tnsr::iJ<DataType, 3, Frame> expected_d_shift{};
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      expected_d_shift.get(k, i) =
          expected_f_4 * expected_x_minus_center.get(i) *
          expected_x_minus_center.get(k) * expected_one_over_r_squared;
      if (i == k) {
        expected_d_shift.get(k, i) += expected_f_3;
      }
    }
  }
  CHECK_ITERABLE_APPROX(d_shift, expected_d_shift);

  tnsr::ii<DataType, 3, Frame> expected_spatial_metric{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      expected_spatial_metric.get(i, j) =
          (expected_spatial_metric_rr - expected_f_0) *
          expected_x_minus_center.get(i) * expected_x_minus_center.get(j) *
          expected_one_over_r_squared;
      if (i == j) {
        expected_spatial_metric.get(i, j) += expected_f_0;
      }
    }
  }
  CHECK_ITERABLE_APPROX(spatial_metric, expected_spatial_metric);

  tnsr::ijj<DataType, 3, Frame> expected_d_spatial_metric{};
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        expected_d_spatial_metric.get(k, i, j) =
            expected_f_2 * expected_x_minus_center.get(i) *
            expected_x_minus_center.get(j) * expected_x_minus_center.get(k) *
            expected_one_over_r_cubed;
        if (i == k) {
          expected_d_spatial_metric.get(k, i, j) +=
              expected_f_1 * expected_x_minus_center.get(j) / expected_r;
        }
        if (j == k) {
          expected_d_spatial_metric.get(k, i, j) +=
              expected_f_1 * expected_x_minus_center.get(i) / expected_r;
        }
        if (i == j) {
          expected_d_spatial_metric.get(k, i, j) +=
              expected_d_f_0 * expected_x_minus_center.get(k) / expected_r;
        }
      }
    }
  }
  CHECK_ITERABLE_APPROX(d_spatial_metric, expected_d_spatial_metric);

  const auto expected_det_and_inverse_spatial_metric =
      determinant_and_inverse(expected_spatial_metric);
  const auto expected_sqrt_det_spatial_metric =
      sqrt(get(expected_det_and_inverse_spatial_metric.first));
  CHECK_ITERABLE_APPROX(get(sqrt_det_spatial_metric),
                        expected_sqrt_det_spatial_metric);

  const auto& expected_inverse_spatial_metric =
      expected_det_and_inverse_spatial_metric.second;
  CHECK_ITERABLE_APPROX(inverse_spatial_metric,
                        expected_inverse_spatial_metric);

  const auto expected_extrinsic_curvature = gr::extrinsic_curvature(
      expected_lapse, expected_shift, expected_d_shift, expected_spatial_metric,
      tnsr::ii<DataType, 3, Frame>(zero), expected_d_spatial_metric);
  CHECK_ITERABLE_APPROX(extrinsic_curvature, expected_extrinsic_curvature);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Gr.HarmonicSchwarzschild",
    "[PointwiseFunctions][Unit]") {
  test_copy_and_move();
  test_serialize();
  test_construct_from_options();

  test_tag_retrieval<Frame::Inertial>(DataVector(5));
  test_tag_retrieval<Frame::Inertial>(0.0);
  test_tag_retrieval<Frame::Grid>(DataVector(5));
  test_tag_retrieval<Frame::Grid>(0.0);

  test_computed_quantities<Frame::Inertial>(DataVector(5));
  test_computed_quantities<Frame::Inertial>(0.0);
  test_computed_quantities<Frame::Grid>(DataVector(5));
  test_computed_quantities<Frame::Grid>(0.0);
}

// [[OutputRegex, Mass must be non-negative]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Gr.HarmonicSchwarzschildMass",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  gr::Solutions::HarmonicSchwarzschild solution(-1.0, {{0.0, 0.0, 0.0}});
}

// [[OutputRegex, In string:.*At line 2 column 9:.Value -0.5 is below the lower
// bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Gr.HarmonicSchwarzschildOptM",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  TestHelpers::test_creation<gr::Solutions::HarmonicSchwarzschild>(
      "Mass: -0.5\n"
      "Center: [1.0,3.0,2.0]");
}
