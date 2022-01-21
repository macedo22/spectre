// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/Systems/Ccz4/ATilde.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/TimeDerivative.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

// Test first order CCZ4 with flat space
void test_minkowski() {
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;

  // Setup solution
  gr::Solutions::Minkowski<SpatialDim> solution{};

  // Setup grid
  const size_t num_points_1d = 4;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  Mesh<SpatialDim> mesh{num_points_1d, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  const size_t num_points_3d = num_points_1d * num_points_1d * num_points_1d;
  const DataVector used_for_size =
      DataVector(num_points_3d, std::numeric_limits<double>::signaling_NaN());
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate solution
  const auto minkowski_vars = solution.variables(
      x, t, typename gr::Solutions::Minkowski<SpatialDim>::tags<DataVector>{});

  // Get ingredients for computing arguments to Ccz4::TimeDerivative
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<SpatialDim, FrameType, DataVector>>(
          minkowski_vars);
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  const auto d_det_spatial_metric =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<SpatialDim, FrameType, DataVector>>(
          minkowski_vars);
  const auto lapse = get<gr::Tags::Lapse<DataVector>>(minkowski_vars);
  const auto shift =
      get<gr::Tags::Shift<SpatialDim, FrameType, DataVector>>(minkowski_vars);
  const auto& d_shift =
      get<Tags::deriv<gr::Tags::Shift<SpatialDim, FrameType, DataVector>,
                      tmpl::size_t<SpatialDim>, FrameType>>(minkowski_vars);

  // Params
  const double c = 1.0;
  const double cleaning_speed = 1.6;  // e
  const double f = 0.75;
  const double kappa_1 = 0.1;
  const double kappa_2 = 0.3;
  const double kappa_3 = 0.4;
  const double mu = 0.7;
  const double one_over_relaxation_time = 10.0;         // \tau^{-1}
  const bool evolve_shift = true;                       // s
  const bool use_shift_advective_terms = true;
  const bool use_harmonic_slicing_condition = false;
  Scalar<DataVector> slicing_condition(used_for_size);  // g(\alpha)
  if (use_harmonic_slicing_condition) {
    get(slicing_condition) = 1.0;
  } else {
    get(slicing_condition) = 2.0 / get(lapse);
  }

  // Choose free variables \Theta, K_0, b^i, and \eta
  const auto theta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto d_theta =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);

  const auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto d_trace_extrinsic_curvature =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);

  const auto k_0 = trace_extrinsic_curvature;
  const auto d_k_0 = d_trace_extrinsic_curvature;

  const auto b = make_with_value<tnsr::I<DataVector, SpatialDim, FrameType>>(
      used_for_size, 0.0);
  const auto d_b = make_with_value<tnsr::iJ<DataVector, SpatialDim, FrameType>>(
      used_for_size, 0.0);

  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);

  // Compute arguments for Ccz4::TimeDerivative
  Scalar<DataVector> ln_lapse{};
  get(ln_lapse) = log(get(lapse));

  const auto field_a =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);
  const auto d_field_a =
      make_with_value<tnsr::ij<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto& field_b = d_shift;
  const auto d_field_b =
      make_with_value<tnsr::ijK<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  // since spatial_metric = conformal_spatial_metric,
  // conformal factor == 1
  const auto conformal_factor_squared =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  const auto ln_conformal_factor =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);

  const auto a_tilde =
      Ccz4::a_tilde(conformal_factor_squared, spatial_metric,
                    extrinsic_curvature, trace_extrinsic_curvature);
  const auto d_a_tilde =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto& conformal_spatial_metric = spatial_metric;
  const auto d_conformal_spatial_metric =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto inverse_conformal_spatial_metric = inverse_spatial_metric;

  const auto field_d =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);
  const auto d_field_d =
      make_with_value<tnsr::ijkk<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto field_d_up =
      make_with_value<tnsr::iJJ<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto field_p =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);
  const auto d_field_p =
      make_with_value<tnsr::ij<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);
  const auto d_conformal_christoffel_second_kind =
      make_with_value<tnsr::iJkk<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto contracted_conformal_christoffel_second_kind =
      Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);
  const auto d_contracted_conformal_christoffel_second_kind =
      make_with_value<tnsr::iJ<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto& gamma_hat = contracted_conformal_christoffel_second_kind;
  const auto& d_gamma_hat = d_contracted_conformal_christoffel_second_kind;

  // LHS time derivatives of evolved variables: eq 12a - 12m
  tnsr::ii<DataVector, SpatialDim> dt_conformal_spatial_metric_actual(
      used_for_size);
  Scalar<DataVector> dt_ln_lapse_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_shift_actual(used_for_size);
  Scalar<DataVector> dt_ln_conformal_factor_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> dt_a_tilde_actual(used_for_size);
  Scalar<DataVector> dt_trace_extrinsic_curvature_actual(used_for_size);
  Scalar<DataVector> dt_theta_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_gamma_hat_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_b_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_a_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim> dt_field_b_actual(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> dt_field_d_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_p_actual(used_for_size);
  // quantities we need for computing eq 12 - 27
  Scalar<DataVector> conformal_factor_squared_actual(used_for_size);
  Scalar<DataVector> det_conformal_spatial_metric_actual(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_conformal_spatial_metric_actual(
      used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_spatial_metric_actual(used_for_size);
  Scalar<DataVector> lapse_actual(used_for_size);
  Scalar<DataVector> slicing_condition_actual(used_for_size);
  Scalar<DataVector> d_slicing_condition_actual(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_a_tilde_actual(used_for_size);
  // quantities we need for computing eq 12 - 27
  tnsr::ij<DataVector, SpatialDim> a_tilde_times_field_b_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim>
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde_actual(
          used_for_size);
  Scalar<DataVector> contracted_field_b_actual(used_for_size);
  tnsr::ijK<DataVector, SpatialDim> symmetrized_d_field_b_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> contracted_symmetrized_d_field_b_actual(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim> field_b_times_field_d_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> field_d_up_times_a_tilde_actual(
      used_for_size);
  tnsr::ij<DataVector, SpatialDim> conformal_metric_times_field_b_actual(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim>
      conformal_metric_times_symmetrized_d_field_b_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> conformal_metric_times_trace_a_tilde_actual(
      used_for_size);
  tnsr::i<DataVector, SpatialDim> inv_conformal_metric_times_d_a_tilde_actual(
      used_for_size);
  tnsr::I<DataVector, SpatialDim>
      gamma_hat_minus_contracted_conformal_christoffel_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim>
      d_gamma_hat_minus_contracted_conformal_christoffel_actual(used_for_size);
  Scalar<DataVector> k_minus_2_theta_c_actual(used_for_size);
  Scalar<DataVector> k_minus_k0_minus_2_theta_c_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> lapse_times_a_tilde_actual(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> lapse_times_d_a_tilde_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> lapse_times_field_a_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> lapse_times_conformal_spatial_metric_actual(
      used_for_size);
  Scalar<DataVector> lapse_times_slicing_condition_actual(used_for_size);
  Scalar<DataVector>
      lapse_times_ricci_scalar_plus_divergence_z4_constraint_actual(
          used_for_size);
  tnsr::I<DataVector, SpatialDim> shift_times_deriv_gamma_hat_actual(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> inv_tau_times_conformal_metric_actual(
      used_for_size);
  // expressions and identities needed for evolution equations: eq 13 - 27
  Scalar<DataVector> trace_a_tilde_actual(used_for_size);
  tnsr::iJJ<DataVector, SpatialDim> field_d_up_actual(used_for_size);
  tnsr::Ijj<DataVector, SpatialDim> conformal_christoffel_second_kind_actual(
      used_for_size);
  tnsr::iJkk<DataVector, SpatialDim> d_conformal_christoffel_second_kind_actual(
      used_for_size);
  tnsr::Ijj<DataVector, SpatialDim> christoffel_second_kind_actual(
      used_for_size);
  tnsr::ij<DataVector, SpatialDim> spatial_ricci_tensor_buffer_actual(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> spatial_ricci_tensor_actual(used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_grad_lapse_actual(used_for_size);
  Scalar<DataVector> divergence_lapse_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim>
      contracted_conformal_christoffel_second_kind_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim>
      d_contracted_conformal_christoffel_second_kind_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> spatial_z4_constraint_actual(used_for_size);
  Scalar<DataVector> upper_spatial_z4_constraint_buffer_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> upper_spatial_z4_constraint_actual(
      used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_spatial_z4_constraint_actual(
      used_for_size);
  Scalar<DataVector> ricci_scalar_plus_divergence_z4_constraint_actual(
      used_for_size);

  ::Ccz4::TimeDerivative<SpatialDim>::apply(
      make_not_null(&dt_conformal_spatial_metric_actual),
      make_not_null(&dt_ln_lapse_actual), make_not_null(&dt_shift_actual),
      make_not_null(&dt_ln_conformal_factor_actual),
      make_not_null(&dt_a_tilde_actual),
      make_not_null(&dt_trace_extrinsic_curvature_actual),
      make_not_null(&dt_theta_actual), make_not_null(&dt_gamma_hat_actual),
      make_not_null(&dt_b_actual), make_not_null(&dt_field_a_actual),
      make_not_null(&dt_field_b_actual), make_not_null(&dt_field_d_actual),
      make_not_null(&dt_field_p_actual),
      make_not_null(&conformal_factor_squared_actual),
      make_not_null(&det_conformal_spatial_metric_actual),
      make_not_null(&inv_conformal_spatial_metric_actual),
      make_not_null(&inv_spatial_metric_actual), make_not_null(&lapse_actual),
      make_not_null(&slicing_condition_actual),
      make_not_null(&d_slicing_condition_actual),
      make_not_null(&inv_a_tilde_actual),
      make_not_null(&a_tilde_times_field_b_actual),
      make_not_null(
          &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde_actual),
      make_not_null(&contracted_field_b_actual),
      make_not_null(&symmetrized_d_field_b_actual),
      make_not_null(&contracted_symmetrized_d_field_b_actual),
      make_not_null(&field_b_times_field_d_actual),
      make_not_null(&field_d_up_times_a_tilde_actual),
      make_not_null(&conformal_metric_times_field_b_actual),
      make_not_null(&conformal_metric_times_symmetrized_d_field_b_actual),
      make_not_null(&conformal_metric_times_trace_a_tilde_actual),
      make_not_null(&inv_conformal_metric_times_d_a_tilde_actual),
      make_not_null(&gamma_hat_minus_contracted_conformal_christoffel_actual),
      make_not_null(&d_gamma_hat_minus_contracted_conformal_christoffel_actual),
      make_not_null(&k_minus_2_theta_c_actual),
      make_not_null(&k_minus_k0_minus_2_theta_c_actual),
      make_not_null(&lapse_times_a_tilde_actual),
      make_not_null(&lapse_times_d_a_tilde_actual),
      make_not_null(&lapse_times_field_a_actual),
      make_not_null(&lapse_times_conformal_spatial_metric_actual),
      make_not_null(&lapse_times_slicing_condition_actual),
      make_not_null(
          &lapse_times_ricci_scalar_plus_divergence_z4_constraint_actual),
      make_not_null(&shift_times_deriv_gamma_hat_actual),
      make_not_null(&inv_tau_times_conformal_metric_actual),
      make_not_null(&trace_a_tilde_actual), make_not_null(&field_d_up_actual),
      make_not_null(&conformal_christoffel_second_kind_actual),
      make_not_null(&d_conformal_christoffel_second_kind_actual),
      make_not_null(&christoffel_second_kind_actual),
      make_not_null(&spatial_ricci_tensor_buffer_actual),
      make_not_null(&spatial_ricci_tensor_actual),
      make_not_null(&grad_grad_lapse_actual),
      make_not_null(&divergence_lapse_actual),
      make_not_null(&contracted_conformal_christoffel_second_kind_actual),
      make_not_null(&d_contracted_conformal_christoffel_second_kind_actual),
      make_not_null(&spatial_z4_constraint_actual),
      make_not_null(&upper_spatial_z4_constraint_buffer_actual),
      make_not_null(&upper_spatial_z4_constraint_actual),
      make_not_null(&grad_spatial_z4_constraint_actual),
      make_not_null(&ricci_scalar_plus_divergence_z4_constraint_actual), c,
      cleaning_speed, eta, f, k_0, d_k_0, kappa_1, kappa_2, kappa_3, mu,
      one_over_relaxation_time, evolve_shift, use_shift_advective_terms,
      use_harmonic_slicing_condition, conformal_spatial_metric, ln_lapse, shift,
      ln_conformal_factor, a_tilde, trace_extrinsic_curvature, theta, gamma_hat,
      b, field_a, field_b, field_d, field_p, d_a_tilde,
      d_trace_extrinsic_curvature, d_theta, d_gamma_hat, d_b, d_field_a,
      d_field_b, d_field_d, d_field_p);

  // Check that all time derivatives are 0
  for (auto& component : dt_conformal_spatial_metric_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_ln_lapse_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_shift_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_ln_conformal_factor_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_a_tilde_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_trace_extrinsic_curvature_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_theta_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_gamma_hat_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_b_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_field_a_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_field_b_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_field_d_actual) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_field_p_actual) {
    CHECK(component == 0.0);
  }
}

void test_kerrschild() {
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;

  // Setup solution
  const double mass = 2.;
  const std::array<double, 3> spin{{0.3, 0.5, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  // Setup grid
  const size_t num_points_1d = 6;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  Mesh<SpatialDim> mesh{num_points_1d, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  const size_t num_points_3d = num_points_1d * num_points_1d * num_points_1d;
  const DataVector used_for_size =
      DataVector(num_points_3d, std::numeric_limits<double>::signaling_NaN());
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();
  // Evaliuate analytic solution
  const auto kerrschild_vars = solution.variables(
      x, t, typename gr::Solutions::KerrSchild::tags<DataVector>{});

  // Get ingredients for computing arguments to Ccz4::TimeDerivative
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<SpatialDim, FrameType, DataVector>>(
          kerrschild_vars);
  const auto& d_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<SpatialDim>,
                      tmpl::size_t<SpatialDim>, FrameType>>(kerrschild_vars);
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  const auto d_det_spatial_metric =
      get<gr::Tags::DerivDetSpatialMetric<SpatialDim, FrameType>>(
          solution.variables(
              x, t,
              tmpl::list<
                  gr::Tags::DerivDetSpatialMetric<SpatialDim, FrameType>>{}));
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<SpatialDim>>>(kerrschild_vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<SpatialDim, FrameType, DataVector>>(
          kerrschild_vars);
  const auto lapse = get<gr::Tags::Lapse<DataVector>>(kerrschild_vars);
  const auto& d_lapse =
      get<Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                      FrameType>>(kerrschild_vars);
  const auto& shift =
      get<gr::Tags::Shift<SpatialDim, FrameType, DataVector>>(kerrschild_vars);
  const auto& d_shift =
      get<Tags::deriv<gr::Tags::Shift<SpatialDim, FrameType, DataVector>,
                      tmpl::size_t<SpatialDim>, FrameType>>(kerrschild_vars);

  // Params
  const double c = 1.0;
  const double cleaning_speed = 1.6;  // e
  const double f = 0.75;
  const double kappa_1 = 0.1;
  const double kappa_2 = 0.3;
  const double kappa_3 = 0.4;
  const double mu = 0.7;
  const double one_over_relaxation_time = 10.0;         // \tau^{-1}
  const bool evolve_shift = true;                       // s
  const bool use_shift_advective_terms = true;
  const bool use_harmonic_slicing_condition = false;
  Scalar<DataVector> slicing_condition(used_for_size);  // g(\alpha)
  if (use_harmonic_slicing_condition) {
    get(slicing_condition) = 1.0;
  } else {
    get(slicing_condition) = 2.0 / get(lapse);
  }

  // Choose free variables \Theta, K_0, b^i, and \eta
  const auto theta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto d_theta =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);

  const auto extrinsic_curvature =
      gr::extrinsic_curvature(lapse, shift, d_shift, spatial_metric,
                              dt_spatial_metric, d_spatial_metric);

  Scalar<DataVector> trace_extrinsic_curvature(used_for_size);
  get(trace_extrinsic_curvature) = 0.0;
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = 0; j < SpatialDim; j++) {
      get(trace_extrinsic_curvature) +=
          extrinsic_curvature.get(i, j) * inverse_spatial_metric.get(i, j);
    }
  }
  const auto d_trace_extrinsic_curvature = partial_derivative(
      trace_extrinsic_curvature, mesh, coord_map.inv_jacobian(x_logical));

  // Solve eq (4g) for K_0, where \partial_t \alpha = 0:
  //   \partial_t \alpha =
  //       -\alpha^2 g(\alpha) (K - K_0 - 2 \Theta) +
  //       \Beta^k \partial_k \alpha
  //   K_0 = -(
  //       (\Beta^k \partial_k \alpha) / (\alpha^2 * g(\alpha)) -
  //       K + 2 \Theta);
  Scalar<DataVector> k_0(used_for_size);
  get(k_0) = get<0>(shift) * get<0>(d_lapse);
  for (size_t k = 1; k < SpatialDim; k++) {
    get(k_0) += shift.get(k) * d_lapse.get(k);
  }
  get(k_0) = -((get(k_0) / (square(get(lapse)) * get(slicing_condition))) -
               get(trace_extrinsic_curvature) + 2.0 * get(theta));
  const auto d_k_0 =
      partial_derivative(k_0, mesh, coord_map.inv_jacobian(x_logical));

  // Solve eq (4h) for b^i, where \partial_t \Beta^i = 0:
  //   \partial_t \Beta^i = f b + \Beta^k \partial_k \Beta^i
  tnsr::I<DataVector, SpatialDim, FrameType> b{};
  if (use_shift_advective_terms) {
    //   0 = f b + \Beta^k \partial_k \Beta^i
    //   b = -(\Beta^k \partial_k \Beta^i) / f
    for (size_t i = 0; i < SpatialDim; i++) {
      b.get(i) = -shift.get(0) * d_shift.get(0, i);
      for (size_t k = 1; k < SpatialDim; k++) {
        b.get(i) -= shift.get(k) * d_shift.get(k, i);
      }
      b.get(i) /= f;
    }
  } else {
    //   0 = f b
    //   b = 0
    for (size_t i = 0; i < SpatialDim; i++) {
      b.get(i) = 0.0;
    }
  }
  const auto d_b =
      partial_derivative(b, mesh, coord_map.inv_jacobian(x_logical));

  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);

  // Compute arguments for Ccz4::TimeDerivative
  Scalar<DataVector> ln_lapse{};
  get(ln_lapse) = log(get(lapse));

  tnsr::i<DataVector, SpatialDim, FrameType> field_a{};
  for (size_t i = 0; i < SpatialDim; i++) {
    field_a.get(i) = d_lapse.get(i) / get(lapse);
  }

  const auto d_d_lapse =
      partial_derivative(d_lapse, mesh, coord_map.inv_jacobian(x_logical));
  // eq:
  //   A_i = \partial_i \alpha / \alpha
  //   \partial_i A_j =
  //       ((\partial_i (\partial_j \alpha)) \alpha -
  //         \partial_j \alpha \partial_i \alpha) / \alpha^2
  tnsr::ij<DataVector, SpatialDim, FrameType> d_field_a{};
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = 0; j < SpatialDim; j++) {
      d_field_a.get(i, j) =
          (d_d_lapse.get(i, j) * get(lapse) - d_lapse.get(j) * d_lapse.get(i)) /
          square(get(lapse));
    }
  }

  const auto& field_b = d_shift;
  const auto d_field_b =
      partial_derivative(field_b, mesh, coord_map.inv_jacobian(x_logical));

  const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = square(conformal_factor);
  Scalar<DataVector> ln_conformal_factor{};
  get(ln_conformal_factor) = log(conformal_factor);

  const auto a_tilde =
      Ccz4::a_tilde(conformal_factor_squared, spatial_metric,
                    extrinsic_curvature, trace_extrinsic_curvature);
  const auto d_a_tilde =
      partial_derivative(a_tilde, mesh, coord_map.inv_jacobian(x_logical));

  tnsr::ii<DataVector, SpatialDim, FrameType> conformal_spatial_metric{};
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = i; j < SpatialDim; j++) {
      conformal_spatial_metric.get(i, j) =
          get(conformal_factor_squared) * spatial_metric.get(i, j);
    }
  }

  tnsr::ijj<DataVector, SpatialDim, FrameType> d_conformal_spatial_metric{};
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        d_conformal_spatial_metric.get(k, i, j) =
            get(conformal_factor_squared) * d_spatial_metric.get(k, i, j) -
            pow<4>(get(conformal_factor_squared)) *
                d_det_spatial_metric.get(k) * spatial_metric.get(i, j) / 3.;
      }
    }
  }

  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(conformal_spatial_metric).second;

  tnsr::ijj<DataVector, SpatialDim, FrameType> field_d{};
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        field_d.get(k, i, j) = 0.5 * d_conformal_spatial_metric.get(k, i, j);
      }
    }
  }
  const auto d_field_d =
      partial_derivative(field_d, mesh, coord_map.inv_jacobian(x_logical));

  auto field_d_up = gr::deriv_inverse_spatial_metric(
      inverse_conformal_spatial_metric, field_d);
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        field_d_up.get(k, i, j) *= -1.0;
      }
    }
  }

  tnsr::i<DataVector, SpatialDim, FrameType> field_p{};
  for (size_t i = 0; i < SpatialDim; i++) {
    field_p.get(i) =
        -d_det_spatial_metric.get(i) / (6. * get(det_spatial_metric));
  }
  const auto d_field_p =
      partial_derivative(field_p, mesh, coord_map.inv_jacobian(x_logical));

  const auto d_conformal_christoffel_second_kind =
      Ccz4::deriv_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d, d_field_d, field_d_up);
  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);

  const auto contracted_conformal_christoffel_second_kind =
      Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);
  const auto d_contracted_conformal_christoffel_second_kind =
      Ccz4::deriv_contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d_up,
          conformal_christoffel_second_kind,
          d_conformal_christoffel_second_kind);

  const auto& gamma_hat = contracted_conformal_christoffel_second_kind;
  const auto& d_gamma_hat = d_contracted_conformal_christoffel_second_kind;

  // LHS time derivatives of evolved variables: eq 12a - 12m
  tnsr::ii<DataVector, SpatialDim> dt_conformal_spatial_metric_actual(
      used_for_size);
  Scalar<DataVector> dt_ln_lapse_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_shift_actual(used_for_size);
  Scalar<DataVector> dt_ln_conformal_factor_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> dt_a_tilde_actual(used_for_size);
  Scalar<DataVector> dt_trace_extrinsic_curvature_actual(used_for_size);
  Scalar<DataVector> dt_theta_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_gamma_hat_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_b_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_a_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim> dt_field_b_actual(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> dt_field_d_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_p_actual(used_for_size);
  // quantities we need for computing eq 12 - 27
  Scalar<DataVector> conformal_factor_squared_actual(used_for_size);
  Scalar<DataVector> det_conformal_spatial_metric_actual(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_conformal_spatial_metric_actual(
      used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_spatial_metric_actual(used_for_size);
  Scalar<DataVector> lapse_actual(used_for_size);
  Scalar<DataVector> slicing_condition_actual(used_for_size);
  Scalar<DataVector> d_slicing_condition_actual(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_a_tilde_actual(used_for_size);
  // quantities we need for computing eq 12 - 27
  tnsr::ij<DataVector, SpatialDim> a_tilde_times_field_b_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim>
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde_actual(
          used_for_size);
  Scalar<DataVector> contracted_field_b_actual(used_for_size);
  tnsr::ijK<DataVector, SpatialDim> symmetrized_d_field_b_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> contracted_symmetrized_d_field_b_actual(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim> field_b_times_field_d_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> field_d_up_times_a_tilde_actual(
      used_for_size);
  tnsr::ij<DataVector, SpatialDim> conformal_metric_times_field_b_actual(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim>
      conformal_metric_times_symmetrized_d_field_b_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> conformal_metric_times_trace_a_tilde_actual(
      used_for_size);
  tnsr::i<DataVector, SpatialDim> inv_conformal_metric_times_d_a_tilde_actual(
      used_for_size);
  tnsr::I<DataVector, SpatialDim>
      gamma_hat_minus_contracted_conformal_christoffel_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim>
      d_gamma_hat_minus_contracted_conformal_christoffel_actual(used_for_size);
  Scalar<DataVector> k_minus_2_theta_c_actual(used_for_size);
  Scalar<DataVector> k_minus_k0_minus_2_theta_c_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> lapse_times_a_tilde_actual(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> lapse_times_d_a_tilde_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> lapse_times_field_a_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> lapse_times_conformal_spatial_metric_actual(
      used_for_size);
  Scalar<DataVector> lapse_times_slicing_condition_actual(used_for_size);
  Scalar<DataVector>
      lapse_times_ricci_scalar_plus_divergence_z4_constraint_actual(
          used_for_size);
  tnsr::I<DataVector, SpatialDim> shift_times_deriv_gamma_hat_actual(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> inv_tau_times_conformal_metric_actual(
      used_for_size);
  // expressions and identities needed for evolution equations: eq 13 - 27
  Scalar<DataVector> trace_a_tilde_actual(used_for_size);
  tnsr::iJJ<DataVector, SpatialDim> field_d_up_actual(used_for_size);
  tnsr::Ijj<DataVector, SpatialDim> conformal_christoffel_second_kind_actual(
      used_for_size);
  tnsr::iJkk<DataVector, SpatialDim> d_conformal_christoffel_second_kind_actual(
      used_for_size);
  tnsr::Ijj<DataVector, SpatialDim> christoffel_second_kind_actual(
      used_for_size);
  tnsr::ij<DataVector, SpatialDim> spatial_ricci_tensor_buffer_actual(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> spatial_ricci_tensor_actual(used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_grad_lapse_actual(used_for_size);
  Scalar<DataVector> divergence_lapse_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim>
      contracted_conformal_christoffel_second_kind_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim>
      d_contracted_conformal_christoffel_second_kind_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> spatial_z4_constraint_actual(used_for_size);
  Scalar<DataVector> upper_spatial_z4_constraint_buffer_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> upper_spatial_z4_constraint_actual(
      used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_spatial_z4_constraint_actual(
      used_for_size);
  Scalar<DataVector> ricci_scalar_plus_divergence_z4_constraint_actual(
      used_for_size);

  ::Ccz4::TimeDerivative<SpatialDim>::apply(
      make_not_null(&dt_conformal_spatial_metric_actual),
      make_not_null(&dt_ln_lapse_actual), make_not_null(&dt_shift_actual),
      make_not_null(&dt_ln_conformal_factor_actual),
      make_not_null(&dt_a_tilde_actual),
      make_not_null(&dt_trace_extrinsic_curvature_actual),
      make_not_null(&dt_theta_actual), make_not_null(&dt_gamma_hat_actual),
      make_not_null(&dt_b_actual), make_not_null(&dt_field_a_actual),
      make_not_null(&dt_field_b_actual), make_not_null(&dt_field_d_actual),
      make_not_null(&dt_field_p_actual),
      make_not_null(&conformal_factor_squared_actual),
      make_not_null(&det_conformal_spatial_metric_actual),
      make_not_null(&inv_conformal_spatial_metric_actual),
      make_not_null(&inv_spatial_metric_actual), make_not_null(&lapse_actual),
      make_not_null(&slicing_condition_actual),
      make_not_null(&d_slicing_condition_actual),
      make_not_null(&inv_a_tilde_actual),
      make_not_null(&a_tilde_times_field_b_actual),
      make_not_null(
          &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde_actual),
      make_not_null(&contracted_field_b_actual),
      make_not_null(&symmetrized_d_field_b_actual),
      make_not_null(&contracted_symmetrized_d_field_b_actual),
      make_not_null(&field_b_times_field_d_actual),
      make_not_null(&field_d_up_times_a_tilde_actual),
      make_not_null(&conformal_metric_times_field_b_actual),
      make_not_null(&conformal_metric_times_symmetrized_d_field_b_actual),
      make_not_null(&conformal_metric_times_trace_a_tilde_actual),
      make_not_null(&inv_conformal_metric_times_d_a_tilde_actual),
      make_not_null(&gamma_hat_minus_contracted_conformal_christoffel_actual),
      make_not_null(&d_gamma_hat_minus_contracted_conformal_christoffel_actual),
      make_not_null(&k_minus_2_theta_c_actual),
      make_not_null(&k_minus_k0_minus_2_theta_c_actual),
      make_not_null(&lapse_times_a_tilde_actual),
      make_not_null(&lapse_times_d_a_tilde_actual),
      make_not_null(&lapse_times_field_a_actual),
      make_not_null(&lapse_times_conformal_spatial_metric_actual),
      make_not_null(&lapse_times_slicing_condition_actual),
      make_not_null(
          &lapse_times_ricci_scalar_plus_divergence_z4_constraint_actual),
      make_not_null(&shift_times_deriv_gamma_hat_actual),
      make_not_null(&inv_tau_times_conformal_metric_actual),
      make_not_null(&trace_a_tilde_actual), make_not_null(&field_d_up_actual),
      make_not_null(&conformal_christoffel_second_kind_actual),
      make_not_null(&d_conformal_christoffel_second_kind_actual),
      make_not_null(&christoffel_second_kind_actual),
      make_not_null(&spatial_ricci_tensor_buffer_actual),
      make_not_null(&spatial_ricci_tensor_actual),
      make_not_null(&grad_grad_lapse_actual),
      make_not_null(&divergence_lapse_actual),
      make_not_null(&contracted_conformal_christoffel_second_kind_actual),
      make_not_null(&d_contracted_conformal_christoffel_second_kind_actual),
      make_not_null(&spatial_z4_constraint_actual),
      make_not_null(&upper_spatial_z4_constraint_buffer_actual),
      make_not_null(&upper_spatial_z4_constraint_actual),
      make_not_null(&grad_spatial_z4_constraint_actual),
      make_not_null(&ricci_scalar_plus_divergence_z4_constraint_actual), c,
      cleaning_speed, eta, f, k_0, d_k_0, kappa_1, kappa_2, kappa_3, mu,
      one_over_relaxation_time, evolve_shift, use_shift_advective_terms,
      use_harmonic_slicing_condition, conformal_spatial_metric, ln_lapse, shift,
      ln_conformal_factor, a_tilde, trace_extrinsic_curvature, theta, gamma_hat,
      b, field_a, field_b, field_d, field_p, d_a_tilde,
      d_trace_extrinsic_curvature, d_theta, d_gamma_hat, d_b, d_field_a,
      d_field_b, d_field_d, d_field_p);

  const auto zero = DataVector(used_for_size.size(), 0.0);

  // Check time derivatives eq (12a) - (12m)
  for (auto& component : dt_conformal_spatial_metric_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_ln_lapse_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_shift_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_ln_conformal_factor_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  Approx approx_12e = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_a_tilde_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12e);
  }
  Approx approx_12f = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_trace_extrinsic_curvature_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12f);
  }
  Approx approx_12g = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_theta_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12g);
  }
  Approx approx_12h = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_gamma_hat_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12h);
  }
  // dt_b will not be 0 for KerrSchild if evolve_shift == true and
  // use_shift_advective_terms == true
  tnsr::i<DataVector, SpatialDim, FrameType> dt_b_expected(used_for_size);
  if (not evolve_shift) {
    for (auto& component : dt_b_expected) {
      component = 0.0;
    }
  } else {
    for (size_t i = 0; i < SpatialDim; i++) {
      dt_b_expected.get(i) = -get(eta) * b.get(i);
    }
    if (use_shift_advective_terms) {
      for (size_t i = 0; i < SpatialDim; i++) {
        dt_b_expected.get(i) +=
            shift.get(0) * d_b.get(0, i) - shift.get(0) * d_gamma_hat.get(0, i);
        for (size_t k = 1; k < SpatialDim; k++) {
          dt_b_expected.get(i) += shift.get(k) * d_b.get(k, i) -
                                  shift.get(k) * d_gamma_hat.get(k, i);
        }
      }
    }
  }
  Approx approx_12i = Approx::custom().epsilon(1e-11).scale(1.0);
  for (size_t i = 0; i < SpatialDim; i++) {
    CHECK_ITERABLE_CUSTOM_APPROX(dt_b_actual.get(i), dt_b_expected.get(i),
                                 approx_12i);
  }
  Approx approx_12j = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_field_a_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12j);
  }
  // dt_field_b will not be 0 for KerrSchild if evolve_shift == true and
  // use_shift_advective_terms == true
  tnsr::iJ<DataVector, SpatialDim, FrameType> dt_field_b_expected(
      used_for_size);
  if (not evolve_shift) {
    for (auto& component : dt_field_b_expected) {
      component = 0.0;
    }
  } else {
    for (size_t k = 0; k < SpatialDim; k++) {
      for (size_t i = 0; i < SpatialDim; i++) {
        dt_field_b_expected.get(k, i) =
            f * d_b.get(k, i) + field_b.get(k, 0) * field_b.get(0, i);
        for (size_t l = 1; l < SpatialDim; l++) {
          dt_field_b_expected.get(k, i) +=
              field_b.get(k, l) * field_b.get(l, i);
        }
      }
    }
    if (use_shift_advective_terms) {
      for (size_t k = 0; k < SpatialDim; k++) {
        for (size_t i = 0; i < SpatialDim; i++) {
          dt_field_b_expected.get(k, i) +=
              shift.get(0) * d_field_b.get(0, k, i);
          for (size_t l = 1; l < SpatialDim; l++) {
            dt_field_b_expected.get(k, i) +=
                shift.get(l) * d_field_b.get(l, k, i);
          }
        }
      }
    }
  }
  Approx approx_12k = Approx::custom().epsilon(1e-11).scale(1.0);
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      CHECK_ITERABLE_CUSTOM_APPROX(dt_field_b_actual.get(k, i),
                                   dt_field_b_expected.get(k, i), approx_12k);
    }
  }
  Approx approx_12l = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_field_d_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12l);
  }
  Approx approx_12m = Approx::custom().epsilon(1e-12).scale(1.0);
  for (auto& component : dt_field_p_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12m);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.TimeDerivative",
                  "[Unit][Evolution]") {
  test_minkowski();
  test_kerrschild();
}
