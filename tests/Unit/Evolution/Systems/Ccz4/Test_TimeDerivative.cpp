// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <iostream>  // TODO: remove
#include <limits>
#include <string>

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
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TimeDerivative.hpp"
//#include "Framework/CheckWithRandomValues.hpp"
//#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
//#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t Dim, typename Generator>
void test_impl(const gsl::not_null<Generator*> generator,
               const DataVector& used_for_size) {
  // const auto lapse = TestHelpers::gr::random_lapse(generator, used_for_size);
  // const auto shift = TestHelpers::gr::random_shift<Dim>(generator,
  // used_for_size); const spatial_metric =
  //     TestHelpers::gr::random_spatial_metric<Dim>(generator, used_for_size);

  tnsr::ii<DataVector, Dim> dt_conformal_spatial_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> dt_ln_lapse =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::I<DataVector, Dim> dt_shift =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> dt_ln_conformal_factor =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::ii<DataVector, Dim> dt_a_tilde =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> dt_trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  Scalar<DataVector> dt_theta =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::I<DataVector, Dim> dt_gamma_hat =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::I<DataVector, Dim> dt_b =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::i<DataVector, Dim> dt_field_a =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::iJ<DataVector, Dim> dt_field_b =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ijj<DataVector, Dim> dt_field_d =
      make_with_value<tnsr::ijj<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::i<DataVector, Dim> dt_field_p =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::I<DataVector, Dim> gamma_hat_minus_contracted_conformal_christoffel =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::iJ<DataVector, Dim> d_gamma_hat_minus_contracted_conformal_christoffel =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> k_minus_2_theta_c =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  Scalar<DataVector> k_minus_k0_minus_2_theta_c =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  Scalar<DataVector> contracted_field_b =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::ij<DataVector, Dim> conformal_metric_times_field_b =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ijk<DataVector, Dim> conformal_metric_times_symmetrized_d_field_b =
      make_with_value<tnsr::ijk<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ij<DataVector, Dim> a_tilde_times_field_b =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> lapse_times_ricci_scalar_plus_divergence_z4_constraint =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::ii<DataVector, Dim> conformal_metric_times_trace_a_tilde =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ii<DataVector, Dim> lapse_times_a_tilde =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::i<DataVector, Dim> field_d_up_times_a_tilde =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ijj<DataVector, Dim> lapse_times_d_a_tilde =
      make_with_value<tnsr::ijj<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::i<DataVector, Dim> inv_conformal_metric_times_d_a_tilde =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ii<DataVector, Dim>
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde =
          make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::i<DataVector, Dim> lapse_times_field_a =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::I<DataVector, Dim> shift_times_deriv_gamma_hat =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ii<DataVector, Dim> inv_tau_times_conformal_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> lapse_times_slicing_condition =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  Scalar<DataVector> conformal_factor_squared =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  Scalar<DataVector> det_conformal_spatial_metric =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::II<DataVector, Dim> inv_conformal_spatial_metric =
      make_with_value<tnsr::II<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::II<DataVector, Dim> inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> lapse =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::ii<DataVector, Dim> lapse_times_conformal_spatial_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> d_slicing_condition =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::II<DataVector, Dim> inv_a_tilde =
      make_with_value<tnsr::II<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ijK<DataVector, Dim> symmetrized_d_field_b =
      make_with_value<tnsr::ijK<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::i<DataVector, Dim> contracted_symmetrized_d_field_b =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ijk<DataVector, Dim> field_b_times_field_d =
      make_with_value<tnsr::ijk<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> trace_a_tilde =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::iJJ<DataVector, Dim> field_d_up =
      make_with_value<tnsr::iJJ<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::Ijj<DataVector, Dim> conformal_christoffel_second_kind =
      make_with_value<tnsr::Ijj<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::iJkk<DataVector, Dim> d_conformal_christoffel_second_kind =
      make_with_value<tnsr::iJkk<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::Ijj<DataVector, Dim> christoffel_second_kind =
      make_with_value<tnsr::Ijj<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ij<DataVector, Dim> spatial_ricci_tensor_buffer =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ii<DataVector, Dim> spatial_ricci_tensor =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ij<DataVector, Dim> grad_grad_lapse =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> divergence_lapse =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::I<DataVector, Dim> contracted_conformal_christoffel_second_kind =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::iJ<DataVector, Dim> d_contracted_conformal_christoffel_second_kind =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::i<DataVector, Dim> spatial_z4_constraint =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> upper_spatial_z4_constraint_buffer =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  tnsr::I<DataVector, Dim> upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  tnsr::ij<DataVector, Dim> grad_spatial_z4_constraint =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  Scalar<DataVector> ricci_scalar_plus_divergence_z4_constraint =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto c = make_with_value<double>(used_for_size, 0.0);
  const auto cleaning_speed = make_with_value<double>(used_for_size, 0.0);
  const auto eta = make_with_value<double>(used_for_size, 0.0);
  const auto f = make_with_value<double>(used_for_size, 0.0);
  const auto slicing_condition =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto k_0 = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto d_k_0 =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  const auto kappa_1 = make_with_value<double>(used_for_size, 0.0);
  const auto kappa_2 = make_with_value<double>(used_for_size, 0.0);
  const auto kappa_3 = make_with_value<double>(used_for_size, 0.0);
  const auto mu = make_with_value<double>(used_for_size, 0.0);
  const auto s = make_with_value<double>(used_for_size, 0.0);
  const auto one_over_relaxation_time =
      make_with_value<double>(used_for_size, 0.0);
  const auto conformal_spatial_metric =
      TestHelpers::gr::random_spatial_metric<Dim>(generator, used_for_size);
  const auto ln_lapse = TestHelpers::gr::random_lapse(generator, used_for_size);
  const auto shift =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  const auto ln_conformal_factor =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto a_tilde =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  const auto trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto theta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto gamma_hat =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  const auto b = make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  const auto field_a =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  const auto field_b =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  const auto field_d =
      make_with_value<tnsr::ijj<DataVector, Dim>>(used_for_size, 0.0);
  const auto field_p =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  //   const auto d_conformal_spatial_metric =
  //       make_with_value<tnsr::ijj<DataVector, Dim>>(used_for_size, 0.0);
  //   const auto d_ln_lapse =
  //       make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  //   const auto d_shift =
  //       make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  //   const auto d_ln_conformal_factor =
  //       make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_a_tilde =
      make_with_value<tnsr::ijj<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_trace_extrinsic_curvature =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_theta =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_gamma_hat =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_b =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_field_a =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_field_b =
      make_with_value<tnsr::ijK<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_field_d =
      make_with_value<tnsr::ijkk<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_field_p =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);

  ::Ccz4::TimeDerivative<Dim>::apply(
      make_not_null(&dt_conformal_spatial_metric), make_not_null(&dt_ln_lapse),
      make_not_null(&dt_shift), make_not_null(&dt_ln_conformal_factor),
      make_not_null(&dt_a_tilde), make_not_null(&dt_trace_extrinsic_curvature),
      make_not_null(&dt_theta), make_not_null(&dt_gamma_hat),
      make_not_null(&dt_b), make_not_null(&dt_field_a),
      make_not_null(&dt_field_b), make_not_null(&dt_field_d),
      make_not_null(&dt_field_p),
      make_not_null(&gamma_hat_minus_contracted_conformal_christoffel),
      make_not_null(&d_gamma_hat_minus_contracted_conformal_christoffel),
      make_not_null(&k_minus_2_theta_c),
      make_not_null(&k_minus_k0_minus_2_theta_c),
      make_not_null(&contracted_field_b),
      make_not_null(&conformal_metric_times_field_b),
      make_not_null(&conformal_metric_times_symmetrized_d_field_b),
      make_not_null(&a_tilde_times_field_b),
      make_not_null(&lapse_times_ricci_scalar_plus_divergence_z4_constraint),
      make_not_null(&conformal_metric_times_trace_a_tilde),
      make_not_null(&lapse_times_a_tilde),
      make_not_null(&field_d_up_times_a_tilde),
      make_not_null(&lapse_times_d_a_tilde),
      make_not_null(&inv_conformal_metric_times_d_a_tilde),
      make_not_null(
          &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde),
      make_not_null(&lapse_times_field_a),
      make_not_null(&shift_times_deriv_gamma_hat),
      make_not_null(&inv_tau_times_conformal_metric),
      make_not_null(&lapse_times_slicing_condition),
      make_not_null(&conformal_factor_squared),
      make_not_null(&det_conformal_spatial_metric),
      make_not_null(&inv_conformal_spatial_metric),
      make_not_null(&inv_spatial_metric), make_not_null(&lapse),
      make_not_null(&lapse_times_conformal_spatial_metric),
      make_not_null(&d_slicing_condition), make_not_null(&inv_a_tilde),
      make_not_null(&symmetrized_d_field_b),
      make_not_null(&contracted_symmetrized_d_field_b),
      make_not_null(&field_b_times_field_d), make_not_null(&trace_a_tilde),
      make_not_null(&field_d_up),
      make_not_null(&conformal_christoffel_second_kind),
      make_not_null(&d_conformal_christoffel_second_kind),
      make_not_null(&christoffel_second_kind),
      make_not_null(&spatial_ricci_tensor_buffer),
      make_not_null(&spatial_ricci_tensor), make_not_null(&grad_grad_lapse),
      make_not_null(&divergence_lapse),
      make_not_null(&contracted_conformal_christoffel_second_kind),
      make_not_null(&d_contracted_conformal_christoffel_second_kind),
      make_not_null(&spatial_z4_constraint),
      make_not_null(&upper_spatial_z4_constraint_buffer),
      make_not_null(&upper_spatial_z4_constraint),
      make_not_null(&grad_spatial_z4_constraint),
      make_not_null(&ricci_scalar_plus_divergence_z4_constraint), c,
      cleaning_speed, eta, f, slicing_condition, k_0, d_k_0, kappa_1, kappa_2,
      kappa_3, mu, s, one_over_relaxation_time, conformal_spatial_metric,
      ln_lapse, shift, ln_conformal_factor, a_tilde, trace_extrinsic_curvature,
      theta, gamma_hat, b, field_a, field_b, field_d, field_p,
      //   d_conformal_spatial_metric, d_ln_lapse, d_shift,
      //   d_ln_conformal_factor,
      d_a_tilde, d_trace_extrinsic_curvature, d_theta, d_gamma_hat, d_b,
      d_field_a, d_field_b, d_field_d, d_field_p);
}

template <typename Generator>
void test(const gsl::not_null<Generator*> generator,
          const DataVector& used_for_size) {
  //   test_impl<1>(generator, used_for_size);
  //   test_impl<2>(generator, used_for_size);
  test_impl<3>(generator, used_for_size);
}

// template <typename Frame, typename DataType>
// tnsr::I<DataType, 3, Frame> spatial_coords(const DataType& used_for_size) {
//   auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
//   get<0>(x) = 1.32;
//   get<1>(x) = 0.82;
//   get<2>(x) = 1.24;
//   return x;
// }

void test_minkowski() {
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;

  // Evaluate solution
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

  const auto minkowski_vars = solution.variables(
      x, t, typename gr::Solutions::Minkowski<SpatialDim>::tags<DataVector>{});

  // Get ingredients for computing arguments to Ccz4::TimeDerivative
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<SpatialDim, FrameType, DataVector>>(
          minkowski_vars);
  const auto& d_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<SpatialDim>,
                      tmpl::size_t<SpatialDim>, FrameType>>(minkowski_vars);
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  const auto d_det_spatial_metric =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<SpatialDim>>>(minkowski_vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<SpatialDim, FrameType, DataVector>>(
          minkowski_vars);
  const auto lapse = get<gr::Tags::Lapse<DataVector>>(minkowski_vars);
  const auto& d_lapse =
      get<Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                      FrameType>>(minkowski_vars);
  const auto shift =
      get<gr::Tags::Shift<SpatialDim, FrameType, DataVector>>(minkowski_vars);
  const auto& d_shift =
      get<Tags::deriv<gr::Tags::Shift<SpatialDim, FrameType, DataVector>,
                      tmpl::size_t<SpatialDim>, FrameType>>(minkowski_vars);
  const auto& field_b = d_shift;
  const auto d_field_b =
      make_with_value<tnsr::ijK<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);
  const auto b = make_with_value<tnsr::I<DataVector, SpatialDim, FrameType>>(
      used_for_size, 0.0);
  const auto d_b = make_with_value<tnsr::iJ<DataVector, SpatialDim, FrameType>>(
      used_for_size, 0.0);

  // Compute arguments for Ccz4::TimeDerivative
  Scalar<DataVector> ln_lapse{};
  get(ln_lapse) = log(get(lapse));

  tnsr::i<DataVector, SpatialDim, FrameType> field_a{};
  for (size_t i = 0; i < SpatialDim; i++) {
    field_a.get(i) = d_lapse.get(i) / get(lapse);
  }
  const auto d_field_a =
      make_with_value<tnsr::ij<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  // TODO : remove this conformal_factor if we don't need it
  const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = square(conformal_factor);

  Scalar<DataVector> ln_conformal_factor{};
  get(ln_conformal_factor) = log(conformal_factor);

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
      make_with_value<tnsr::ijkk<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  auto field_d_up = gr::deriv_inverse_spatial_metric(
      inverse_conformal_spatial_metric, field_d);
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        field_d_up.get(k, i, j) *= -1.0;
      }
    }
  }

  //   using field_d_tag = Ccz4::Tags::FieldD<SpatialDim, FrameType,
  //   DataVector>; Variables<tmpl::list<field_d_tag>>
  //   field_d_var(num_points_3d); get<field_d_tag>(field_d_var) = field_d;
  //   const auto d_field_d_var = partial_derivatives<tmpl::list<field_d_tag>>(
  //       field_d_var, mesh, coord_map.inv_jacobian(x_logical));
  //   const auto& d_field_d =
  //       get<Tags::deriv<field_d_tag, tmpl::size_t<SpatialDim>, FrameType>>(
  //           d_field_d_var);

  const auto d_conformal_christoffel_second_kind =
      Ccz4::deriv_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d, d_field_d, field_d_up);

  tnsr::i<DataVector, SpatialDim, FrameType> field_p{};
  for (size_t i = 0; i < SpatialDim; i++) {
    field_p.get(i) =
        -d_det_spatial_metric.get(i) / (6. * get(det_spatial_metric));
  }

  const auto d_field_p =
      make_with_value<tnsr::ij<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  //   using field_p_tag = Ccz4::Tags::FieldP<SpatialDim, FrameType,
  //   DataVector>; Variables<tmpl::list<field_p_tag>>
  //   field_p_var(num_points_3d); get<field_p_tag>(field_p_var) = field_p;
  //   const auto d_field_p_var = partial_derivatives<tmpl::list<field_p_tag>>(
  //       field_p_var, mesh, coord_map.inv_jacobian(x_logical));
  //   const auto& d_field_p =
  //       get<Tags::deriv<field_p_tag, tmpl::size_t<SpatialDim>, FrameType>>(
  //           d_field_p_var);

  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);

  const auto contracted_conformal_christoffel_second_kind =
      Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);
  const auto& gamma_hat = contracted_conformal_christoffel_second_kind;
  const auto d_contracted_conformal_christoffel_second_kind =
      Ccz4::deriv_contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d_up,
          conformal_christoffel_second_kind,
          d_conformal_christoffel_second_kind);
  const auto& d_gamma_hat = d_contracted_conformal_christoffel_second_kind;

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

  const auto d_trace_extrinsic_curvature =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);

  const auto a_tilde =
      Ccz4::a_tilde(conformal_factor_squared, spatial_metric,
                    extrinsic_curvature, trace_extrinsic_curvature);
  const auto d_a_tilde =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto theta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto d_theta =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);

  // params
  const double c = 1.0;
  const double cleaning_speed = 1.6;
  const double eta = 0.5;
  const double f = 0.6;
  const auto slicing_condition =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  const auto k_0 = make_with_value<Scalar<DataVector>>(
      used_for_size, get(trace_extrinsic_curvature)[0]);
  const auto d_k_0 =
      make_with_value<tnsr::i<DataVector, SpatialDim>>(used_for_size, 0.0);
  const double kappa_1 = 0.1;
  const double kappa_2 = 0.3;
  const double kappa_3 = 0.4;
  const double mu = 0.7;
  const double s = 1.0;
  const double one_over_relaxation_time = 10.0;

  // Evolution variables to be filled by Ccz4::TimeDerivative
  tnsr::ii<DataVector, SpatialDim> dt_conformal_spatial_metric(used_for_size);
  Scalar<DataVector> dt_ln_lapse(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_shift(used_for_size);
  Scalar<DataVector> dt_ln_conformal_factor(used_for_size);
  tnsr::ii<DataVector, SpatialDim> dt_a_tilde(used_for_size);
  Scalar<DataVector> dt_trace_extrinsic_curvature(used_for_size);
  Scalar<DataVector> dt_theta(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_gamma_hat(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_b(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_a(used_for_size);
  tnsr::iJ<DataVector, SpatialDim> dt_field_b(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> dt_field_d(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_p(used_for_size);
  // Intermediates to be filled by Ccz4::TimeDerivative
  tnsr::I<DataVector, SpatialDim>
      gamma_hat_minus_contracted_conformal_christoffel(used_for_size);
  tnsr::iJ<DataVector, SpatialDim>
      d_gamma_hat_minus_contracted_conformal_christoffel(used_for_size);
  Scalar<DataVector> k_minus_2_theta_c(used_for_size);
  Scalar<DataVector> k_minus_k0_minus_2_theta_c(used_for_size);
  Scalar<DataVector> contracted_field_b(used_for_size);
  tnsr::ij<DataVector, SpatialDim> conformal_metric_times_field_b(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim>
      conformal_metric_times_symmetrized_d_field_b(used_for_size);
  tnsr::ij<DataVector, SpatialDim> a_tilde_times_field_b(used_for_size);
  Scalar<DataVector> lapse_times_ricci_scalar_plus_divergence_z4_constraint(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> conformal_metric_times_trace_a_tilde(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> lapse_times_a_tilde(used_for_size);
  tnsr::i<DataVector, SpatialDim> field_d_up_times_a_tilde(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> lapse_times_d_a_tilde(used_for_size);
  tnsr::i<DataVector, SpatialDim> inv_conformal_metric_times_d_a_tilde(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim>
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde(
          used_for_size);
  tnsr::i<DataVector, SpatialDim> lapse_times_field_a(used_for_size);
  tnsr::I<DataVector, SpatialDim> shift_times_deriv_gamma_hat(used_for_size);
  tnsr::ii<DataVector, SpatialDim> inv_tau_times_conformal_metric(
      used_for_size);
  Scalar<DataVector> lapse_times_slicing_condition(used_for_size);
  // other things we need for eqs 12 - 27 (TODO : better name)
  //   Scalar<DataVector> conformal_factor_squared(used_for_size);
  Scalar<DataVector> det_conformal_spatial_metric(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_conformal_spatial_metric(
      used_for_size);  // TODO : already computed
  tnsr::II<DataVector, SpatialDim> inv_spatial_metric(
      used_for_size);                               // TODO : already computed
  Scalar<DataVector> lapse_to_fill(used_for_size);  // TODO : already computed
  tnsr::ii<DataVector, SpatialDim> lapse_times_conformal_spatial_metric(
      used_for_size);
  Scalar<DataVector> d_slicing_condition(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_a_tilde(used_for_size);
  tnsr::ijK<DataVector, SpatialDim> symmetrized_d_field_b(used_for_size);
  tnsr::i<DataVector, SpatialDim> contracted_symmetrized_d_field_b(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim> field_b_times_field_d(used_for_size);
  // expressions and identities needed for time derivative eqs (eqs 13 - 27)
  Scalar<DataVector> trace_a_tilde_to_fill(
      used_for_size);  // TODO : already computed
  tnsr::iJJ<DataVector, SpatialDim> field_d_up_to_fill(
      used_for_size);  // TODO : already computed
  tnsr::Ijj<DataVector, SpatialDim> conformal_christoffel_second_kind_to_fill(
      used_for_size);  // TODO : already computed
  tnsr::iJkk<DataVector, SpatialDim>
      d_conformal_christoffel_second_kind_to_fill(
          used_for_size);  // TODO : already computed
  tnsr::Ijj<DataVector, SpatialDim> christoffel_second_kind(used_for_size);
  tnsr::ij<DataVector, SpatialDim> spatial_ricci_tensor_buffer(used_for_size);
  tnsr::ii<DataVector, SpatialDim> spatial_ricci_tensor(used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_grad_lapse(used_for_size);
  Scalar<DataVector> divergence_lapse(used_for_size);
  tnsr::I<DataVector, SpatialDim>
      contracted_conformal_christoffel_second_kind_to_fill(
          used_for_size);  // TODO : already computed
  tnsr::iJ<DataVector, SpatialDim>
      d_contracted_conformal_christoffel_second_kind_to_fill(
          used_for_size);  // TODO : already computed
  tnsr::i<DataVector, SpatialDim> spatial_z4_constraint(used_for_size);
  Scalar<DataVector> upper_spatial_z4_constraint_buffer(used_for_size);
  tnsr::I<DataVector, SpatialDim> upper_spatial_z4_constraint(used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_spatial_z4_constraint(used_for_size);
  Scalar<DataVector> ricci_scalar_plus_divergence_z4_constraint(used_for_size);

  ::Ccz4::TimeDerivative<SpatialDim>::apply(
      make_not_null(&dt_conformal_spatial_metric), make_not_null(&dt_ln_lapse),
      make_not_null(&dt_shift), make_not_null(&dt_ln_conformal_factor),
      make_not_null(&dt_a_tilde), make_not_null(&dt_trace_extrinsic_curvature),
      make_not_null(&dt_theta), make_not_null(&dt_gamma_hat),
      make_not_null(&dt_b), make_not_null(&dt_field_a),
      make_not_null(&dt_field_b), make_not_null(&dt_field_d),
      make_not_null(&dt_field_p),
      make_not_null(&gamma_hat_minus_contracted_conformal_christoffel),
      make_not_null(&d_gamma_hat_minus_contracted_conformal_christoffel),
      make_not_null(&k_minus_2_theta_c),
      make_not_null(&k_minus_k0_minus_2_theta_c),
      make_not_null(&contracted_field_b),
      make_not_null(&conformal_metric_times_field_b),
      make_not_null(&conformal_metric_times_symmetrized_d_field_b),
      make_not_null(&a_tilde_times_field_b),
      make_not_null(&lapse_times_ricci_scalar_plus_divergence_z4_constraint),
      make_not_null(&conformal_metric_times_trace_a_tilde),
      make_not_null(&lapse_times_a_tilde),
      make_not_null(&field_d_up_times_a_tilde),
      make_not_null(&lapse_times_d_a_tilde),
      make_not_null(&inv_conformal_metric_times_d_a_tilde),
      make_not_null(
          &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde),
      make_not_null(&lapse_times_field_a),
      make_not_null(&shift_times_deriv_gamma_hat),
      make_not_null(&inv_tau_times_conformal_metric),
      make_not_null(&lapse_times_slicing_condition),
      make_not_null(&conformal_factor_squared),
      make_not_null(&det_conformal_spatial_metric),
      make_not_null(&inv_conformal_spatial_metric),
      make_not_null(&inv_spatial_metric), make_not_null(&lapse_to_fill),
      make_not_null(&lapse_times_conformal_spatial_metric),
      make_not_null(&d_slicing_condition), make_not_null(&inv_a_tilde),
      make_not_null(&symmetrized_d_field_b),
      make_not_null(&contracted_symmetrized_d_field_b),
      make_not_null(&field_b_times_field_d),
      make_not_null(&trace_a_tilde_to_fill), make_not_null(&field_d_up_to_fill),
      make_not_null(&conformal_christoffel_second_kind_to_fill),
      make_not_null(&d_conformal_christoffel_second_kind_to_fill),
      make_not_null(&christoffel_second_kind),
      make_not_null(&spatial_ricci_tensor_buffer),
      make_not_null(&spatial_ricci_tensor), make_not_null(&grad_grad_lapse),
      make_not_null(&divergence_lapse),
      make_not_null(&contracted_conformal_christoffel_second_kind_to_fill),
      make_not_null(&d_contracted_conformal_christoffel_second_kind_to_fill),
      make_not_null(&spatial_z4_constraint),
      make_not_null(&upper_spatial_z4_constraint_buffer),
      make_not_null(&upper_spatial_z4_constraint),
      make_not_null(&grad_spatial_z4_constraint),
      make_not_null(&ricci_scalar_plus_divergence_z4_constraint), c,
      cleaning_speed, eta, f, slicing_condition, k_0, d_k_0, kappa_1, kappa_2,
      kappa_3, mu, s, one_over_relaxation_time, conformal_spatial_metric,
      ln_lapse, shift, ln_conformal_factor, a_tilde, trace_extrinsic_curvature,
      theta, gamma_hat, b, field_a, field_b, field_d, field_p,
      //   d_conformal_spatial_metric, d_ln_lapse, d_shift,
      //   d_ln_conformal_factor,
      d_a_tilde, d_trace_extrinsic_curvature, d_theta, d_gamma_hat, d_b,
      d_field_a, d_field_b, d_field_d, d_field_p);

  // Check that all time derivatives are 0
  for (auto& component : dt_conformal_spatial_metric) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_ln_lapse) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_shift) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_ln_conformal_factor) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_a_tilde) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_trace_extrinsic_curvature) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_theta) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_gamma_hat) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_b) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_field_a) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_field_b) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_field_d) {
    CHECK(component == 0.0);
  }
  for (auto& component : dt_field_p) {
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
  const auto shift =
      get<gr::Tags::Shift<SpatialDim, FrameType, DataVector>>(kerrschild_vars);
  const auto& d_shift =
      get<Tags::deriv<gr::Tags::Shift<SpatialDim, FrameType, DataVector>,
                      tmpl::size_t<SpatialDim>, FrameType>>(kerrschild_vars);
  //   std::cout << "d_lapse : " << d_lapse << std::endl;
  //   std::cout << "d_shift : " << d_shift << std::endl;
  const auto& field_b = d_shift;
  using field_b_tag = Ccz4::Tags::FieldB<SpatialDim, FrameType, DataVector>;
  Variables<tmpl::list<field_b_tag>> field_b_var(num_points_3d);
  get<field_b_tag>(field_b_var) = field_b;
  const auto d_field_b_var = partial_derivatives<tmpl::list<field_b_tag>>(
      field_b_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_field_b =
      get<Tags::deriv<field_b_tag, tmpl::size_t<SpatialDim>, FrameType>>(
          d_field_b_var);

  // eq:
  //   dt_shift = f * b + shift * d_shift ---> need dt_shift? set dt_shift to 0?
  //   b = (dt_shift - shift * d_shift) / f = (-shift * d_shift) / f
  //   const auto b = make_with_value<tnsr::I<DataVector, SpatialDim,
  //   FrameType>>(
  //       used_for_size, 0.0);
  const double f = 0.6;
  tnsr::I<DataVector, SpatialDim, FrameType> b{};
  for (size_t i = 0; i < SpatialDim; i++) {
    b.get(i) = -shift.get(0) * d_shift.get(0, i) / f;
    for (size_t k = 1; k < SpatialDim; k++) {
      // assuming initial dt_shift == 0.0
      b.get(i) -= shift.get(k) * d_shift.get(k, i) / f;
    }
  }
  // eq:
  //   dt_shift = f * b + shift * d_shift ---> need dt_shift? set dt_shift to 0?
  //   b^i = (dt_shift - shift * d_shift) / f = (-shift^k * d_shift_k^i) / f
  //   d_b_j^i = (-shift^k * d_d_shift_jk^i - d_shift_j^k * d_shift_k^i) / f
  //
  //   using b_tag = Ccz4::Tags::FieldB<SpatialDim, FrameType, DataVector>;
  //   Variables<tmpl::list<b_tag>> b_var(num_points_3d);
  //   get<b_tag>(b_var) = b;
  //   const auto d_b_var = partial_derivatives<tmpl::list<b_tag>>(
  //       b_var, mesh, coord_map.inv_jacobian(x_logical));
  //   const auto& d_b =
  //       get<Tags::deriv<b_tag, tmpl::size_t<SpatialDim>,
  //       FrameType>>(d_b_var);
  tnsr::iJ<DataVector, SpatialDim, FrameType> d_b{};
  for (size_t j = 0; j < SpatialDim; j++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      d_b.get(j, i) = -shift.get(0) * d_field_b.get(j, 0, i) -
                      field_b.get(j, 0) * field_b.get(0, i);
      for (size_t k = 1; k < SpatialDim; k++) {
        // assuming initial dt_shift == 0.0
        d_b.get(j, i) -= shift.get(0) * d_field_b.get(j, 0, i) +
                         field_b.get(j, 0) * field_b.get(0, i);
      }
      d_b.get(j, i) /= f;
    }
  }

  // Compute arguments for Ccz4::TimeDerivative
  Scalar<DataVector> ln_lapse{};
  get(ln_lapse) = log(get(lapse));

  tnsr::i<DataVector, SpatialDim, FrameType> field_a{};
  for (size_t i = 0; i < SpatialDim; i++) {
    field_a.get(i) = d_lapse.get(i) / get(lapse);
  }
  using d_lapse_tag = Tags::deriv<gr::Tags::Lapse<DataVector>,
                                  tmpl::size_t<SpatialDim>, FrameType>;
  Variables<tmpl::list<d_lapse_tag>> d_lapse_var(num_points_3d);
  get<d_lapse_tag>(d_lapse_var) = d_lapse;
  const auto d_d_lapse_var = partial_derivatives<tmpl::list<d_lapse_tag>>(
      d_lapse_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_d_lapse =
      get<Tags::deriv<d_lapse_tag, tmpl::size_t<SpatialDim>, FrameType>>(
          d_d_lapse_var);
  // eq:
  //   field_a_i = d_lapse_i / lapse
  //   d_field_a_ji = (d_d_lapse_ji * lapse - d_lapse_i * d_lapse_j) / lapse^2
  tnsr::ij<DataVector, SpatialDim, FrameType> d_field_a{};
  for (size_t j = 0; j < SpatialDim; j++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      d_field_a.get(j, i) =
          (d_d_lapse.get(j, i) * get(lapse) - d_lapse.get(i) * d_lapse.get(j)) /
          square(get(lapse));
    }
  }

  // TODO : remove this conformal_factor if we don't need it
  const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = square(conformal_factor);

  Scalar<DataVector> ln_conformal_factor{};
  get(ln_conformal_factor) = log(conformal_factor);

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
  using field_d_tag = Ccz4::Tags::FieldD<SpatialDim, FrameType, DataVector>;
  Variables<tmpl::list<field_d_tag>> field_d_var(num_points_3d);
  get<field_d_tag>(field_d_var) = field_d;
  const auto d_field_d_var = partial_derivatives<tmpl::list<field_d_tag>>(
      field_d_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_field_d =
      get<Tags::deriv<field_d_tag, tmpl::size_t<SpatialDim>, FrameType>>(
          d_field_d_var);

  auto field_d_up = gr::deriv_inverse_spatial_metric(
      inverse_conformal_spatial_metric, field_d);
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        field_d_up.get(k, i, j) *= -1.0;
      }
    }
  }

  const auto d_conformal_christoffel_second_kind =
      Ccz4::deriv_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d, d_field_d, field_d_up);

  tnsr::i<DataVector, SpatialDim, FrameType> field_p{};
  for (size_t i = 0; i < SpatialDim; i++) {
    field_p.get(i) =
        -d_det_spatial_metric.get(i) / (6. * get(det_spatial_metric));
  }
  using field_p_tag = Ccz4::Tags::FieldP<SpatialDim, FrameType, DataVector>;
  Variables<tmpl::list<field_p_tag>> field_p_var(num_points_3d);
  get<field_p_tag>(field_p_var) = field_p;
  const auto d_field_p_var = partial_derivatives<tmpl::list<field_p_tag>>(
      field_p_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_field_p =
      get<Tags::deriv<field_p_tag, tmpl::size_t<SpatialDim>, FrameType>>(
          d_field_p_var);

  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);

  const auto contracted_conformal_christoffel_second_kind =
      Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);
  //   const auto christoffel_second_kind =
  //       gr::christoffel_second_kind(d_spatial_metric,
  //       inverse_spatial_metric);
  //   using christoffel_second_kind_tag =
  //       gr::Tags::SpatialChristoffelSecondKind<SpatialDim, FrameType,
  //       DataVector>;
  //   Variables<tmpl::list<christoffel_second_kind_tag>>
  //       christoffel_second_kind_var(num_points_3d);
  //   get<christoffel_second_kind_tag>(christoffel_second_kind_var) =
  //       christoffel_second_kind;
  //   const auto d_christoffel_second_kind_var =
  //       partial_derivatives<tmpl::list<christoffel_second_kind_tag>>(
  //           christoffel_second_kind_var, mesh,
  //           coord_map.inv_jacobian(x_logical));
  //   const auto& d_christoffel_second_kind =
  //       get<Tags::deriv<christoffel_second_kind_tag,
  //       tmpl::size_t<SpatialDim>,
  //                       FrameType>>(d_christoffel_second_kind_var);
  //   const auto spatial_ricci_tensor_kerr =
  //       gr::ricci_tensor(christoffel_second_kind, d_christoffel_second_kind);
  //   const auto spacetime_normal_one_form =
  //       gr::spacetime_normal_one_form(lapse);
  // TODO : need to actually compute this...
  const auto& gamma_hat = contracted_conformal_christoffel_second_kind;
  const auto d_contracted_conformal_christoffel_second_kind =
      Ccz4::deriv_contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d_up,
          conformal_christoffel_second_kind,
          d_conformal_christoffel_second_kind);
  const auto& d_gamma_hat = d_contracted_conformal_christoffel_second_kind;

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
  using trace_extrinsic_curvature_tag =
      gr::Tags::TraceExtrinsicCurvature<DataVector>;
  Variables<tmpl::list<trace_extrinsic_curvature_tag>>
      trace_extrinsic_curvature_var(num_points_3d);
  get<trace_extrinsic_curvature_tag>(trace_extrinsic_curvature_var) =
      trace_extrinsic_curvature;
  const auto d_trace_extrinsic_curvature_var =
      partial_derivatives<tmpl::list<trace_extrinsic_curvature_tag>>(
          trace_extrinsic_curvature_var, mesh,
          coord_map.inv_jacobian(x_logical));
  const auto& d_trace_extrinsic_curvature =
      get<Tags::deriv<trace_extrinsic_curvature_tag, tmpl::size_t<SpatialDim>,
                      FrameType>>(d_trace_extrinsic_curvature_var);

  const auto a_tilde =
      Ccz4::a_tilde(conformal_factor_squared, spatial_metric,
                    extrinsic_curvature, trace_extrinsic_curvature);
  using a_tilde_tag = Ccz4::Tags::ATilde<SpatialDim, FrameType, DataVector>;
  Variables<tmpl::list<a_tilde_tag>> a_tilde_var(num_points_3d);
  get<a_tilde_tag>(a_tilde_var) = a_tilde;
  const auto d_a_tilde_var = partial_derivatives<tmpl::list<a_tilde_tag>>(
      a_tilde_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_a_tilde =
      get<Tags::deriv<a_tilde_tag, tmpl::size_t<SpatialDim>, FrameType>>(
          d_a_tilde_var);

  //   const auto theta = make_with_value<Scalar<DataVector>>(used_for_size,
  //   0.0); const auto d_theta =
  //       make_with_value<tnsr::i<DataVector, SpatialDim,
  //       FrameType>>(used_for_size,
  //                                                                 0.0);

  // TODO : revisit these values after reading Rezolla
  // params
  const double c = 1.0;
  const double cleaning_speed = 1.6;
  const double eta = 0.5;
  //   const double f = 0.6;
  //   auto slicing_condition =
  //       make_with_value<Scalar<DataVector>>(used_for_size, 2.0);
  //   get(slicing_condition) /= get(lapse);
  const auto slicing_condition =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  get(ln_lapse) = log(get(lapse));
  const auto& k_0 = trace_extrinsic_curvature;
  const auto& d_k_0 = d_trace_extrinsic_curvature;
  const double kappa_1 = 0.1;
  const double kappa_2 = 0.3;
  const double kappa_3 = 0.4;
  const double mu = 0.7;
  const double s = 1.0;
  const double one_over_relaxation_time = 10.0;

  //   const auto theta = make_with_value<Scalar<DataVector>>(used_for_size,
  //   0.0);
  // eq (let dt_ln_lapse = 0.0):
  //   theta = ((shift^k * A_k) / (lapse * g(lapse)) + K - K_0) / (2c)
  //   auto theta = make_with_value<Scalar<DataVector>>(
  //       used_for_size, get<0>(shift) * get<0>(field_a));
  Scalar<DataVector> theta(used_for_size);
  get(theta) = get<0>(shift) * get<0>(field_a);
  for (size_t k = 1; k < SpatialDim; k++) {
    get(theta) += shift.get(k) * field_a.get(k);
  }
  get(theta) = ((get(theta) / (get(lapse) * get(slicing_condition))) -
                get(trace_extrinsic_curvature) + get(k_0)) /
               (-2.0 * c);
  using theta_tag = Ccz4::Tags::Theta<DataVector>;
  Variables<tmpl::list<theta_tag>> theta_var(num_points_3d);
  get<theta_tag>(theta_var) = theta;
  const auto d_theta_var = partial_derivatives<tmpl::list<theta_tag>>(
      theta_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_theta =
      get<Tags::deriv<theta_tag, tmpl::size_t<SpatialDim>, FrameType>>(
          d_theta_var);

  // Evolution variables to be filled by Ccz4::TimeDerivative
  tnsr::ii<DataVector, SpatialDim> dt_conformal_spatial_metric(used_for_size);
  Scalar<DataVector> dt_ln_lapse(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_shift(used_for_size);
  Scalar<DataVector> dt_ln_conformal_factor(used_for_size);
  tnsr::ii<DataVector, SpatialDim> dt_a_tilde(used_for_size);
  Scalar<DataVector> dt_trace_extrinsic_curvature(used_for_size);
  Scalar<DataVector> dt_theta(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_gamma_hat(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_b(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_a(used_for_size);
  tnsr::iJ<DataVector, SpatialDim> dt_field_b(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> dt_field_d(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_p(used_for_size);
  // Intermediates to be filled by Ccz4::TimeDerivative
  tnsr::I<DataVector, SpatialDim>
      gamma_hat_minus_contracted_conformal_christoffel(used_for_size);
  tnsr::iJ<DataVector, SpatialDim>
      d_gamma_hat_minus_contracted_conformal_christoffel(used_for_size);
  Scalar<DataVector> k_minus_2_theta_c(used_for_size);
  Scalar<DataVector> k_minus_k0_minus_2_theta_c(used_for_size);
  Scalar<DataVector> contracted_field_b(used_for_size);
  tnsr::ij<DataVector, SpatialDim> conformal_metric_times_field_b(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim>
      conformal_metric_times_symmetrized_d_field_b(used_for_size);
  tnsr::ij<DataVector, SpatialDim> a_tilde_times_field_b(used_for_size);
  Scalar<DataVector> lapse_times_ricci_scalar_plus_divergence_z4_constraint(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> conformal_metric_times_trace_a_tilde(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> lapse_times_a_tilde(used_for_size);
  tnsr::i<DataVector, SpatialDim> field_d_up_times_a_tilde(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> lapse_times_d_a_tilde(used_for_size);
  tnsr::i<DataVector, SpatialDim> inv_conformal_metric_times_d_a_tilde(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim>
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde(
          used_for_size);
  tnsr::i<DataVector, SpatialDim> lapse_times_field_a(used_for_size);
  tnsr::I<DataVector, SpatialDim> shift_times_deriv_gamma_hat(used_for_size);
  tnsr::ii<DataVector, SpatialDim> inv_tau_times_conformal_metric(
      used_for_size);
  Scalar<DataVector> lapse_times_slicing_condition(used_for_size);
  // other things we need for eqs 12 - 27 (TODO : better name)
  //   Scalar<DataVector> conformal_factor_squared(used_for_size);
  Scalar<DataVector> det_conformal_spatial_metric(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_conformal_spatial_metric(
      used_for_size);  // TODO : already computed
  tnsr::II<DataVector, SpatialDim> inv_spatial_metric(
      used_for_size);                               // TODO : already computed
  Scalar<DataVector> lapse_to_fill(used_for_size);  // TODO : already computed
  tnsr::ii<DataVector, SpatialDim> lapse_times_conformal_spatial_metric(
      used_for_size);
  Scalar<DataVector> d_slicing_condition(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_a_tilde(used_for_size);
  tnsr::ijK<DataVector, SpatialDim> symmetrized_d_field_b(used_for_size);
  tnsr::i<DataVector, SpatialDim> contracted_symmetrized_d_field_b(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim> field_b_times_field_d(used_for_size);
  // expressions and identities needed for time derivative eqs (eqs 13 - 27)
  Scalar<DataVector> trace_a_tilde_to_fill(
      used_for_size);  // TODO : already computed
  tnsr::iJJ<DataVector, SpatialDim> field_d_up_to_fill(
      used_for_size);  // TODO : already computed
  tnsr::Ijj<DataVector, SpatialDim> conformal_christoffel_second_kind_to_fill(
      used_for_size);  // TODO : already computed
  tnsr::iJkk<DataVector, SpatialDim>
      d_conformal_christoffel_second_kind_to_fill(
          used_for_size);  // TODO : already computed
  tnsr::Ijj<DataVector, SpatialDim> christoffel_second_kind(used_for_size);
  tnsr::ij<DataVector, SpatialDim> spatial_ricci_tensor_buffer(used_for_size);
  tnsr::ii<DataVector, SpatialDim> spatial_ricci_tensor(used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_grad_lapse(used_for_size);
  Scalar<DataVector> divergence_lapse(used_for_size);
  tnsr::I<DataVector, SpatialDim>
      contracted_conformal_christoffel_second_kind_to_fill(
          used_for_size);  // TODO : already computed
  tnsr::iJ<DataVector, SpatialDim>
      d_contracted_conformal_christoffel_second_kind_to_fill(
          used_for_size);  // TODO : already computed
  tnsr::i<DataVector, SpatialDim> spatial_z4_constraint(used_for_size);
  Scalar<DataVector> upper_spatial_z4_constraint_buffer(used_for_size);
  tnsr::I<DataVector, SpatialDim> upper_spatial_z4_constraint(used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_spatial_z4_constraint(used_for_size);
  Scalar<DataVector> ricci_scalar_plus_divergence_z4_constraint(used_for_size);

  ::Ccz4::TimeDerivative<SpatialDim>::apply(
      make_not_null(&dt_conformal_spatial_metric), make_not_null(&dt_ln_lapse),
      make_not_null(&dt_shift), make_not_null(&dt_ln_conformal_factor),
      make_not_null(&dt_a_tilde), make_not_null(&dt_trace_extrinsic_curvature),
      make_not_null(&dt_theta), make_not_null(&dt_gamma_hat),
      make_not_null(&dt_b), make_not_null(&dt_field_a),
      make_not_null(&dt_field_b), make_not_null(&dt_field_d),
      make_not_null(&dt_field_p),
      make_not_null(&gamma_hat_minus_contracted_conformal_christoffel),
      make_not_null(&d_gamma_hat_minus_contracted_conformal_christoffel),
      make_not_null(&k_minus_2_theta_c),
      make_not_null(&k_minus_k0_minus_2_theta_c),
      make_not_null(&contracted_field_b),
      make_not_null(&conformal_metric_times_field_b),
      make_not_null(&conformal_metric_times_symmetrized_d_field_b),
      make_not_null(&a_tilde_times_field_b),
      make_not_null(&lapse_times_ricci_scalar_plus_divergence_z4_constraint),
      make_not_null(&conformal_metric_times_trace_a_tilde),
      make_not_null(&lapse_times_a_tilde),
      make_not_null(&field_d_up_times_a_tilde),
      make_not_null(&lapse_times_d_a_tilde),
      make_not_null(&inv_conformal_metric_times_d_a_tilde),
      make_not_null(
          &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde),
      make_not_null(&lapse_times_field_a),
      make_not_null(&shift_times_deriv_gamma_hat),
      make_not_null(&inv_tau_times_conformal_metric),
      make_not_null(&lapse_times_slicing_condition),
      make_not_null(&conformal_factor_squared),
      make_not_null(&det_conformal_spatial_metric),
      make_not_null(&inv_conformal_spatial_metric),
      make_not_null(&inv_spatial_metric), make_not_null(&lapse_to_fill),
      make_not_null(&lapse_times_conformal_spatial_metric),
      make_not_null(&d_slicing_condition), make_not_null(&inv_a_tilde),
      make_not_null(&symmetrized_d_field_b),
      make_not_null(&contracted_symmetrized_d_field_b),
      make_not_null(&field_b_times_field_d),
      make_not_null(&trace_a_tilde_to_fill), make_not_null(&field_d_up_to_fill),
      make_not_null(&conformal_christoffel_second_kind_to_fill),
      make_not_null(&d_conformal_christoffel_second_kind_to_fill),
      make_not_null(&christoffel_second_kind),
      make_not_null(&spatial_ricci_tensor_buffer),
      make_not_null(&spatial_ricci_tensor), make_not_null(&grad_grad_lapse),
      make_not_null(&divergence_lapse),
      make_not_null(&contracted_conformal_christoffel_second_kind_to_fill),
      make_not_null(&d_contracted_conformal_christoffel_second_kind_to_fill),
      make_not_null(&spatial_z4_constraint),
      make_not_null(&upper_spatial_z4_constraint_buffer),
      make_not_null(&upper_spatial_z4_constraint),
      make_not_null(&grad_spatial_z4_constraint),
      make_not_null(&ricci_scalar_plus_divergence_z4_constraint), c,
      cleaning_speed, eta, f, slicing_condition, k_0, d_k_0, kappa_1, kappa_2,
      kappa_3, mu, s, one_over_relaxation_time, conformal_spatial_metric,
      ln_lapse, shift, ln_conformal_factor, a_tilde, trace_extrinsic_curvature,
      theta, gamma_hat, b, field_a, field_b, field_d, field_p,
      //   d_conformal_spatial_metric, d_ln_lapse, d_shift,
      //   d_ln_conformal_factor,
      d_a_tilde, d_trace_extrinsic_curvature, d_theta, d_gamma_hat, d_b,
      d_field_a, d_field_b, d_field_d, d_field_p);

  const auto zero = DataVector(used_for_size.size(), 0.0);
  // Check that all time derivatives are 0
  for (auto& component : dt_conformal_spatial_metric) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_ln_lapse) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_shift) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_ln_conformal_factor) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  //   for (auto& component : dt_a_tilde) {
  //     CHECK_ITERABLE_APPROX(component, zero);
  //   }
  //   for (auto& component : dt_trace_extrinsic_curvature) {
  //     CHECK_ITERABLE_APPROX(component, zero);
  //   }
  //   for (auto& component : dt_theta) {
  //     CHECK_ITERABLE_APPROX(component, zero);
  //   }
  //   for (auto& component : dt_gamma_hat) {
  //     CHECK_ITERABLE_APPROX(component, zero);
  //   }
  //   for (auto& component : dt_b) {
  //     CHECK_ITERABLE_APPROX(component, zero);
  //   }
  // TODO : this is a large tolerance, maybe try computing some of the initial
  // derivatives analytically to see if this tolerance can be improved
  Approx approx_12j = Approx::custom().epsilon(1e-7).scale(1.0);
  for (auto& component : dt_field_a) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12j);
  }
  //   for (auto& component : dt_field_b) {
  //     CHECK_ITERABLE_APPROX(component, zero);
  //   }
  //   for (auto& component : dt_field_d) {
  //     CHECK_ITERABLE_APPROX(component, zero);
  //   }
  //   for (auto& component : dt_field_p) {
  //     CHECK_ITERABLE_APPROX(component, zero);
  //   }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.TimeDerivative",
                  "[Unit][Evolution]") {
  //   MAKE_GENERATOR(generator);

  //   test(make_not_null(&generator),
  //        DataVector(5, std::numeric_limits<double>::signaling_NaN()));
  //   test_minkowski();
  test_kerrschild();
}
