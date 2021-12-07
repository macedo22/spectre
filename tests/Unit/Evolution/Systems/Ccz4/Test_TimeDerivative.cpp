// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/TimeDerivative.hpp"
//#include "Framework/CheckWithRandomValues.hpp"
//#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
//#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
//#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <size_t Dim, typename Generator>
void test_impl(const gsl::not_null<Generator*> generator,
               const DataVector& used_for_size) {
  //   const auto lapse = TestHelpers::gr::random_lapse(generator,
  //   used_for_size); const auto shift =
  //   TestHelpers::gr::random_shift<3>(generator, used_for_size); const
  //   spatial_metric =
  //       TestHelpers::gr::random_spatial_metric<3>(generator, used_for_size);

  (void)generator;  // TODO : remove

  auto dt_conformal_spatial_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  auto dt_ln_lapse = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto dt_shift = make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  auto dt_ln_conformal_factor =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto dt_a_tilde =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  auto dt_trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto dt_theta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto dt_gamma_hat =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  auto dt_b = make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  auto dt_field_a =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  auto dt_field_b =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  auto dt_field_d =
      make_with_value<tnsr::ijj<DataVector, Dim>>(used_for_size, 0.0);
  auto dt_field_p =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  auto gamma_hat_minus_contracted_conformal_christoffel =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  auto d_gamma_hat_minus_contracted_conformal_christoffel =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  auto k_minus_2_theta_c =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto k_minus_k0_minus_2_theta_c =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto contracted_field_b =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto conformal_metric_times_field_b =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  auto conformal_metric_times_symmetrized_d_field_b =
      make_with_value<tnsr::ijk<DataVector, Dim>>(used_for_size, 0.0);
  auto a_tilde_times_field_b =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  auto lapse_times_ricci_scalar_plus_divergence_z4_constraint =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto conformal_metric_times_trace_a_tilde =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  auto lapse_times_a_tilde =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  auto field_d_up_times_a_tilde =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  auto lapse_times_d_a_tilde =
      make_with_value<tnsr::ijj<DataVector, Dim>>(used_for_size, 0.0);
  auto inv_conformal_metric_times_d_a_tilde =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  auto a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  auto lapse_times_field_a =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  auto shift_times_deriv_gamma_hat =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  auto inv_tau_times_conformal_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  auto lapse_times_slicing_condition =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto conformal_factor_squared =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto det_conformal_spatial_metric =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto inv_conformal_spatial_metric =
      make_with_value<tnsr::II<DataVector, Dim>>(used_for_size, 0.0);
  auto inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, Dim>>(used_for_size, 0.0);
  auto lapse = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto lapse_times_conformal_spatial_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  auto d_slicing_condition =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto inv_a_tilde =
      make_with_value<tnsr::II<DataVector, Dim>>(used_for_size, 0.0);
  auto symmetrized_d_field_b =
      make_with_value<tnsr::ijK<DataVector, Dim>>(used_for_size, 0.0);
  auto contracted_symmetrized_d_field_b =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  auto field_b_times_field_d =
      make_with_value<tnsr::ijk<DataVector, Dim>>(used_for_size, 0.0);
  auto trace_a_tilde = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto field_d_up =
      make_with_value<tnsr::iJJ<DataVector, Dim>>(used_for_size, 0.0);
  auto conformal_christoffel_second_kind =
      make_with_value<tnsr::Ijj<DataVector, Dim>>(used_for_size, 0.0);
  auto d_conformal_christoffel_second_kind =
      make_with_value<tnsr::iJkk<DataVector, Dim>>(used_for_size, 0.0);
  auto christoffel_second_kind =
      make_with_value<tnsr::Ijj<DataVector, Dim>>(used_for_size, 0.0);
  auto spatial_ricci_tensor_buffer =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  auto spatial_ricci_tensor =
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  auto grad_grad_lapse =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  auto divergence_lapse =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto contracted_conformal_christoffel_second_kind =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  auto d_contracted_conformal_christoffel_second_kind =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  auto spatial_z4_constraint =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  auto upper_spatial_z4_constraint_buffer =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);
  auto grad_spatial_z4_constraint =
      make_with_value<tnsr::ij<DataVector, Dim>>(used_for_size, 0.0);
  auto ricci_scalar_plus_divergence_z4_constraint =
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
      make_with_value<tnsr::ii<DataVector, Dim>>(used_for_size, 0.0);
  const auto ln_lapse = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
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
  const auto d_conformal_spatial_metric =
      make_with_value<tnsr::ijj<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_ln_lapse =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_shift =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.0);
  const auto d_ln_conformal_factor =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);
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
      d_conformal_spatial_metric, d_ln_lapse, d_shift, d_ln_conformal_factor,
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
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.TimeDerivative",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);

  test(make_not_null(&generator),
       DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
