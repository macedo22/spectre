// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CCZ4/TimeDerivative.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim>
void TimeDerivative<Dim>::apply(
    // time derivatives of evolved variables (eqs 12a - 12m)
    const gsl::not_null<tnsr::ij<DataVector, Dim>*> dt_conformal_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> dt_ln_lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_shift,
    const gsl::not_null<Scalar<DataVector>*> dt_ln_conformal_factor,
    const gsl::not_null<tnsr::ij<DataVector, Dim>*> dt_a_tilde,
    const gsl::not_null<Scalar<DataVector>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Scalar<DataVector>*> dt_theta,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_gamma_hat,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_b,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> dt_field_a,
    const gsl::not_null<tnsr::iJ<DataVector, Dim>*> dt_field_b,
    const gsl::not_null<tnsr::ijk<DataVector, Dim>*> dt_field_d,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> dt_field_p,
    // temporary expressions defined by TempTags.hpp
    // TODO (maybe) : reorder the temp quantities here and in TempTags.hpp
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        gamma_hat_minus_contracted_conformal_christoffel,
    const gsl::not_null<Scalar<DataVector>*> k_minus_2_theta_c,
    const gsl::not_null<Scalar<DataVector>*> k_minus_k0_minus_2_theta_c,
    const gsl::not_null<Scalar<DataVector>*> contracted_field_b,
    const gsl::not_null<tnsr::ij<DataVector, Dim>*>
        conformal_metric_times_field_b,
    const gsl::not_null<tnsr::ij<DataVector, Dim>*>
        a_tilde_times_field_b,  // TODO : need to add to TempTags.hpp
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        conformal_metric_times_trace_a_tilde,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_ricci_scalar_plus_divergence_z4_constraint,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        conformal_metric_times_trace_a_tilde,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*> lapse_times_a_tilde,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> field_d_up_times_a_tilde,
    const gsl::not_null<tnsr::ijj<DataVector, Dim>*> lapse_times_d_a_tilde,
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        inv_conformal_metric_times_d_a_tilde,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> lapse_times_field_a,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> shift_times_deriv_gamma_hat,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        inv_tau_times_conformal_metric,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_slicing_condition,
    // other things we need for eqs 12 - 27 (TODO : better name)
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_squared,
    const gsl::not_null<Scalar<DataVector>*>
        det_conformal_spatial_metric,  // 13, TODO : need to add simple tag?
    const gsl::not_null<tnsr::II<DataVector, Dim>*>
        inv_conformal_spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, Dim>*> inv_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        lapse_times_conformal_spatial_metric,
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        d_slicing_condition,  // g'(alpha)
    const gsl::not_null<tnsr::II<DataVector, Dim>*> inv_a_tilde,
    const gsl::not_null<tnsr::ijK<DataVector, Dim>*> symmetrized_d_field_b,
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        contracted_symmetrized_d_field_b,
    const gsl::not_null<tnsr::ijk<DataVector, Dim>*> field_b_times_field_d,
    // expressions and identities needed for time derivative eqs (eqs 13 - 27)
    const gsl::not_null<Scalar<DataVector>*> trace_a_tilde,       // 13
    const gsl::not_null<tnsr::iJJ<DataVector, Dim>*> field_d_up,  // 14
    const gsl::not_null<tnsr::Ijj<DataVector, Dim>*>
        conformal_christoffel_second_kind,  // 15
    const gsl::not_null<tnsr::iJkk<DataVector, Dim>*>
        d_conformal_christoffel_second_kind,  // 16
    const gsl::not_null<tnsr::Ijj<DataVector, Dim>*>
        christoffel_second_kind,  // 17
    const gsl::not_null<tnsr::ij<DataType, Dim, Frame>*>
        spatial_ricci_tensor_buffer,  // buffer needed for 18 -20
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        spatial_ricci_tensor,                                         // 18 - 20
    const gsl::not_null<tnsr::ij<DataVector, Dim>*> grad_grad_lapse,  // 21
    const gsl::not_null<Scalar<DataVector>*> divergence_lapse,        // 22
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        contracted_conformal_christoffel_second_kind,  // 23
    const gsl::not_null<tnsr::iJ<DataVector, Dim>*>
        d_contracted_conformal_christoffel_second_kind,                    // 24
    const gsl::not_null<tnsr::i<DataVector, Dim>*> spatial_z4_constraint,  // 25
    const gsl::not_null<Scalar<DataType>*>
        upper_spatial_z4_constraint_buffer,  // buffer needed for eq 25
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        upper_spatial_z4_constraint,  // 25
    const gsl::not_null<tnsr::ij<DataVector, Dim>*>
        grad_spatial_z4_constraint,  // 26
    const gsl::not_null<Scalar<DataVector>*>
        ricci_scalar_plus_divergence_z4_constraint,  // 27
    // params (TODO: better name?)
    const double c, const double cleaning_speed /*e*/, const double eta,
    const double f, const Scalar<DataVector>& slicing_condition,
    const Scalar<DataVector>& k_0,
    const tnsr::i<DataVector, Dim>&
        d_k_0 /*TODO : how to compute? is k_0 not 0?*/,
    const double kappa_1, const double kappa_2, const double kappa_3,
    const double mu, const double /*TODO : bool ?*/ s,
    const double one_over_relaxation_time,
    // evolved variables
    const tnsr::ij<DataVector, Dim>& conformal_spatial_metric,
    const Scalar<DataVector>& ln_lapse, const tnsr::I<DataVector, Dim>& shift,
    const Scalar<DataVector>& ln_conformal_factor,
    const tnsr::ij<DataVector, Dim>& a_tilde,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& theta, const tnsr::I<DataVector, Dim>& gamma_hat,
    const tnsr::I<DataVector, Dim>& b, const tnsr::i<DataVector, Dim>& field_a,
    const tnsr::iJ<DataVector, Dim>& field_b,
    const tnsr::ijk<DataVector, Dim>& field_d,
    const tnsr::i<DataVector, Dim>& field_p,
    // spatial derivatives of evolved variables
    const tnsr::ij<DataVector, Dim>& d_conformal_spatial_metric,
    const Scalar<DataVector>& d_ln_lapse,
    const tnsr::I<DataVector, Dim>& d_shift,
    const Scalar<DataVector>& d_ln_conformal_factor,
    const tnsr::ij<DataVector, Dim>& d_a_tilde,
    const Scalar<DataVector>& d_trace_extrinsic_curvature,
    const Scalar<DataVector>& d_theta,
    const tnsr::I<DataVector, Dim>& d_gamma_hat,
    const tnsr::I<DataVector, Dim>& d_b,
    const tnsr::i<DataVector, Dim>& d_field_a,
    const tnsr::iJ<DataVector, Dim>& d_field_b,
    const tnsr::ijk<DataVector, Dim>& d_field_d,
    const tnsr::i<DataVector, Dim>& d_field_p) {
  constexpr double one_third = 1.0 / 3.0;
  constexpr double eulers_number = 2.71828182845904523536;

  // Beginning stuff (better name?)
  determinant_and_inverse(det_conformal_spatial_metric,
                          inv_conformal_spatial_metric,
                          conformal_spatial_metric);

  const size_t num_points = get_size(get(ln_conformal_factor));
  for (size_t i = 0; i < num_points; i++) {
    get(*conformal_factor_squared)[i] =
        pow(eulers_number, 2.0 * get(ln_conformal_factor)[i]);
  }

  ::TensorExpressions::evaluate<ti_I, ti_J>(
      inv_spatial_metric,
      (*conformal_factor_squared)() * inv_conformal_spatial_metric(ti_I, ti_J));

  ::TensorExpressions::evaluate<ti_I, ti_J>(
      inv_a_tilde, a_tilde(ti_k, ti_l) * (*inv_spatial_metric)(ti_I, ti_K) *
                       (*inv_spatial_metric)(ti_J, ti_L));

  ::TensorExpressions::evaluate<ti_k, ti_j, ti_I>(
      symmetrized_d_field_b,
      0.5 * d_field_b(ti_k, ti_j, ti_i) + d_field_b(ti_j, ti_k, ti_I));

  ::TensorExpressions::evaluate<ti_k>(
      contracted_symmetrized_d_field_b,
      (*symmetrized_d_field_b)(ti_k, ti_i, ti_I));

  ::TensorExpressions::evaluate<ti_i, ti_j, ti_k>(
      field_b_times_field_d, field_b(ti_i, ti_L) + field_d(ti_j, ti_l, ti_k));

  for (size_t i = 0; i < num_points; i++) {
    get(*lapse)[i] = pow(eulers_number, get(ln_lapse)[i]);
  }

  ::TensorExpressions::evaluate<ti_i, ti_j>(
      lapse_times_conformal_spatial_metric,
      (*lapse)() * conformal_spatial_metric(ti_i, ti_j));

  // if g(\alpha) == 1, then g'(\alpha) == 0
  // if g(\alpha) == 2 / \alpha, then g'(\alpha)  == -2 / \alpha^2
  get(*d_slicing_condition) =
      get(slicing_condition)[0] == 1.0 ? 0.0 : -2.0 / square(get(*lapse));

  // eq 13 - 27

  // eq 13
  ::TensorExpressions::evaluate(
      trace_a_tilde,
      (*inv_conformal_spatial_metric)(ti_I, ti_J) * a_tilde(ti_i, ti_j));

  // eq 14
  // TODO : rebase on develop to get ti_N/n and swap in for ti_L/l here
  ::TensorExpressions::evaluate(
      field_d_up, (*inv_conformal_spatial_metric)(ti_I, ti_L) *
                      (*inv_conformal_spatial_metric)(ti_M, ti_J) *
                      field_d(ti_k, ti_l, ti_m));

  // eq 15
  conformal_christoffel_second_kind(conformal_christoffel_second_kind,
                                    *inv_conformal_spatial_metric, field_d);

  // eq 16
  deriv_conformal_christoffel_second_kind(d_conformal_christoffel_second_kind,
                                          *inv_conformal_spatial_metric,
                                          field_d, d_field_d, *field_d_up);

  // eq 17
  christoffel_second_kind(christoffel_second_kind, conformal_spatial_metric,
                          *inv_conformal_spatial_metric, field_p,
                          *conformal_christoffel_second_kind);

  // eq 18 - 20
  spatial_ricci_tensor(spatial_ricci_tensor, spatial_ricci_tensor_buffer,
                       *christoffel_second_kind,
                       *d_conformal_christoffel_second_kind,
                       conformal_spatial_metric, *inv_conformal_spatial_metric,
                       field_d, *field_d_up, field_p, d_field_p);

  // eq 21
  grad_grad_lapse(grad_grad_lapse, *lapse, *christoffel_second_kind, field_a,
                  d_field_a);

  // eq 22
  divergence_lapse(divergence_lapse, *conformal_factor_squared,
                   *inv_conformal_spatial_metric, *grad_grad_lapse);

  // eq 23
  contracted_conformal_christoffel_second_kind(
      contracted_conformal_christoffel_second_kind,
      *inv_conformal_spatial_metric, *conformal_christoffel_second_kind);

  // eq 24
  deriv_contracted_conformal_christoffel_second_kind(
      deriv_contracted_conformal_christoffel_second_kind,
      *inv_conformal_spatial_metric, *field_d_up,
      *conformal_christoffel_second_kind, *d_conformal_christoffel_second_kind);

  // temp needed for eq 25
  ::TensorExpressions::evaluate<ti_I>(
      gamma_hat(ti_I) - (*contracted_conformal_christoffel_second_kind)(ti_I));

  // eq 25
  spatial_z4_constraint(spatial_z4_constraint, conformal_spatial_metric,
                        *gamma_hat_minus_contracted_conformal_christoffel);

  // eq 25
  upper_spatial_z4_constraint(
      upper_spatial_z4_constraint, upper_spatial_z4_constraint_buffer,
      *conformal_factor_squared,
      *gamma_hat_minus_contracted_conformal_christoffel);

  // eq 26
  grad_spatial_z4_constraint(
      grad_spatial_z4_constraint, *spatial_z4_constraint,
      conformal_spatial_metric, *christoffel_second_kind, field_d,
      *gamma_hat_minus_contracted_conformal_christoffel,
      *d_gamma_hat_minus_contracted_conformal_christoffel);

  // eq 27
  ricci_scalar_plus_divergence_z4_constraint(
      ricci_scalar_plus_divergence_z4_constraint, *conformal_factor_squared,
      *inverse_conformal_spatial_metric, *spatial_ricci_tensor,
      *grad_spatial_z4_constraint);

  // Repeated temporaries in evolution equations
  ::TensorExpressions::evaluate<ti_i, ti_j>(
      a_tilde_times_field_b, a_tilde(ti_k, ti_i) * field_b(ti_j, ti_K));

  ::TensorExpressions::evaluate(
      k_minus_2_theta_c, trace_extrinsic_curvature() - 2.0 * c * theta());

  ::TensorExpressions::evaluate(k_minus_k0_minus_2_theta_c,
                                (*k_minus_2_theta_c)() - k_0());

  ::TensorExpressions::evaluate(contracted_field_b, field_b(ti_k, ti_K));

  ::TensorExpressions::evaluate<ti_i, ti_j>(
      conformal_metric_times_field_b,
      conformal_spatial_metric(ti_k, ti_i) * field_b(ti_j, ti_K));

  ::TensorExpressions::evaluate(
      lapse_times_ricci_scalar_plus_divergence_z4_constraint,
      (*lapse)() * (*ricci_scalar_plus_divergence_z4_constraint)());

  ::TensorExpressions::evaluate<ti_i, ti_j>(
      conformal_metric_times_trace_a_tilde,
      conformal_spatial_metric(ti_i, ti_j) * (*trace_a_tilde)());

  ::TensorExpressions::evaluate<ti_i, ti_j>(lapse_times_a_tilde,
                                            (*lapse)() * a_tilde(ti_i, ti_j));

  ::TensorExpressions::evaluate<ti_k>(
      field_d_up_times_a_tilde,
      (*field_d_up)(ti_k, ti_I, ti_J) * a_tilde(ti_i, ti_j));

  TensorExpressions::evaluate<ti_k, ti_i, ti_j>(
      lapse_times_d_a_tilde, (*lapse()) * d_a_tilde(ti_k, ti_i, ti_j));

  ::TensorExpressions::evaluate<ti_k>(
      inv_conformal_metric_times_d_a_tilde,
      (*inv_conformal_spatial_metric)(ti_I, ti_J) *
          d_a_tilde(ti_k, ti_i, ti_j));

  ::TensorExpressions::evaluate<ti_i, ti_j>(
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde,
      a_tilde(ti_i, ti_j) -
          one_third * (*conformal_metric_times_trace_a_tilde)(ti_i, ti_j));

  ::TensorExpressions::evaluate<ti_k>(lapse_times_field_a,
                                      (*lapse)() * field_a(ti_k));

  ::TensorExpressions::evaluate<ti_I>(shift_times_deriv_gamma_hat,
                                      shift(ti_K) * d_gamma_hat(ti_k, ti_I));

  ::TensorExpressions::evaluate<ti_i, ti_j>(
      inv_tau_times_conformal_metric,
      one_over_relaxation_time * conformal_spatial_metric(ti_i, ti_j));

  // if g(\alpha) == 1, then \alpha g(\alpha) == \alpha
  // if g(\alpha) == 2 / \alpha, then \alpha g(\alpha)  == 2
  get(*lapse_times_slicing_condition) =
      get(slicing_condition)[0] == 1.0 ? get(*lapse) : 2.0;

  // Time derivative computation, eq. (12a) - (12m)

  // eq. (12a) : time derivative of the conformal spatial metric
  // TODO (?): I expect this has repeated calculations for:
  //   -  2.0 * (*lapse)()
  //   -  (*det_conformal_spatial_metric)() - 1.0)
  ::TensorExpressions::evaluate<ti_i, ti_j>(
      dt_conformal_spatial_metric,
      2.0 * shift(ti_K) * field_d(ti_k, ti_i, ti_j) +
          (*conformal_metric_times_field_b)(ti_i, ti_j) +
          (*conformal_metric_times_field_b)(ti_j, ti_i) -
          2.0 * one_third * conformal_spatial_metric(ti_i, ti_j) *
              (*contracted_field_b)() -
          2.0 * (*lapse()) *
              (*a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde)(
                  ti_i, ti_j) -
          (*inv_tau_times_conformal_metric)(ti_i, ti_j) *
              ((*det_conformal_spatial_metric)() - 1.0));

  // eq. (12b) : time derivative of the natural log of the lapse
  ::TensorExpressions::evaluate(
      dt_ln_lapse,
      shift(ti_K) * field_a(ti_k) -
          (*lapse()) * slicing_condition() * (*k_minus_k0_minus_2_theta_c)());

  // eq. (12c) : time derivative of the shift
  // s == 0 or s == 1
  // TODO : have a temporary for terms without s? or implement with s known at
  // compile time so that we can if constexpr and only compile one or the other?
  if (s == 0.0) {
    ::TensorExpressions::evaluate<ti_I>(dt_shift,
                                        s * shift(ti_K) * B(ti_k, ti_I));
  } else {
    ::TensorExpressions::evaluate<ti_I>(
        dt_shift, s * shift(ti_K) * B(ti_k, ti_I) + f * b(ti_I));
  }

  // eq. (12d) : time derivative of the natural log of the conformal
  // factor
  ::TensorExpressions::evaluate(
      dt_ln_conformal_factor,
      shift(ti_K) * field_P(ti_k) +
          one_third * ((*lapse)() * (*trace_extrinsic_curvature)() -
                       (*contracted_field_b)()));

  // TODO : add ATildeTimesFieldB to TempTags.hpp
  // eq. (12e) : time derivative of the trace-free part of the extrinsic
  // curvature
  ::TensorExpressions::evaluate<ti_i, ti_j>(
      dt_a_tilde,
      shift(ti_K) * d_a_tilde(ti_k, ti_i, ti_j) +
          (*conformal_factor_squared)() *
              ((*lapse)() * ((*spatial_ricci_tensor)(ti_i, ti_j) +
                             (*grad_spatial_z4_constraint)(ti_i, ti_j) +
                             (*grad_spatial_z4_constraint)(ti_j, ti_i)) -
               (*grad_grad_lapse)(ti_i, ti_j)) -
          one_third * conformal_spatial_metric(ti_i, ti_j) *
              ((*lapse_times_ricci_scalar_plus_divergence_z4_constraint)() -
               (*divergence_lapse)()) +
          (*a_tilde_times_field_b)(ti_i, ti_j) +
          (*a_tilde_times_field_b)(ti_j, ti_i) -
          2.0 * one_third * (*a_tilde)(ti_i, ti_j) * (*contracted_field_b)() +
          (*lapse_times_a_tilde)(ti_i, ti_j) * (*k_minus_2_theta_c)() -
          2.0 * (*lapse_times_a_tilde)(ti_i, ti_l) *
              (*inv_conformal_spatial_metric)(ti_L, ti_M) *
              a_tilde(ti_m, ti_j) -
          (*inv_tau_times_conformal_metric)(ti_i, ti_j) * (*trace_a_tilde)());

  // eq. (12f) : time derivative of the trace of the extrinsic curvature
  ::TensorExpressions::evaluate(
      dt_trace_extrinsic_curvature,
      shift(ti_K) * d_trace_extrinsic_curvature(ti_k) - (*divergence_lapse)() +
          (*lapse_times_ricci_scalar_plus_divergence_z4_constraint)() +
          (*lapse)() * (trace_extrinsic_curvature() * (*k_minus_2_theta_c)() -
                        3.0 * kappa_1 * (1.0 + kappa_2) * theta()));

  // eq. (12g) : time derivative of the projection of the Z4 four-vector along
  // the normal direction
  ::TensorExpressions::evaluate(
      dt_theta,
      shift(ti_K) * d_theta(ti_k) +
          (*lapse)() *
              (0.5 * square(e) *
                   ((*ricci_scalar_plus_divergence_z4_constraint)() +
                    2.0 * one_third * square(trace_extrinsic_curvature()) -
                    a_tilde(ti_i, ti_j) * (*inv_a_tilde)(ti_I, ti_J)) -
               c * theta() * (*trace_extrinsic_curvature)() -
               (*spatial_z4_constraint)(ti_I)*field_a(ti_i) -
               kappa_1 * (2.0 + kappa_2) * theta()));

  // eq. (12h) : time derivative \hat{\Gamma}^i
  // TODO : have a temporary for terms without s? or implement with s known at
  // compile time so that we can if constexpr and only compile one or the other?
  if (s == 0.0) {
      ::TensorExpressions::evaluate<ti_I>(
        dt_gamma_hat,
        // terms without lapse nor s
        (*shift_times_deriv_gamma_hat)(ti_I) +
            2.0 * one_third *
                (*contracted_conformal_christoffel_second_kind)(ti_I) *
                (*contracted_field_b)() -
            (*contracted_conformal_christoffel_second_kind)(ti_K)*field_b(
                ti_k, ti_I) +
            2.0 * kappa_3 * (*spatial_z4_constraint)(ti_j) *
                (2.0 * one_third * (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                     (*contracted_field_b)() -
                 (*inv_conformal_spatial_metric)(ti_J, ti_K) *
                     field_b(ti_k, ti_I)) +
            // terms with lapse
            2.0 * (*lapse)() *
                (-2.0 * one_third *
                     (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                     d_extrinsic_curvature(ti_j) +
                 (*inv_conformal_spatial_metric)(ti_K, ti_I) * d_theta(ti_k) +
                 (*conformal_christoffel_second_kind)(ti_I, ti_j, ti_k) *
                     (*inv_a_tilde)(ti_J, ti_K) -
                 3.0 * (*inv_a_tilde)(ti_I, ti_J) * field_p(ti_j) -
                 (*inv_conformal_spatial_metric)(ti_K, ti_I) *
                     (theta() * field_a(ti_k) +
                      2.0 * one_third * trace_extrinsic_curvature() *
                          (*spatial_z4_constraint)(ti_k)) -
                 a_tilde(ti_I, ti_J) * field_a(ti_j) -
                 kappa_1 * (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                     (*spatial_z4_constraint)(ti_j));
  } else {
    ::TensorExpressions::evaluate<ti_I>(
        dt_gamma_hat,
        // terms without lapse nor s
        (*shift_times_deriv_gamma_hat)(ti_I) +
            2.0 * one_third *
                (*contracted_conformal_christoffel_second_kind)(ti_I) *
                (*contracted_field_b)() -
            (*contracted_conformal_christoffel_second_kind)(ti_K)*field_b(
                ti_k, ti_I) +
            2.0 * kappa_3 * (*spatial_z4_constraint)(ti_j) *
                (2.0 * one_third * (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                     (*contracted_field_b)() -
                 (*inv_conformal_spatial_metric)(ti_J, ti_K) *
                     field_b(ti_k, ti_I)) +
            // terms with lapse
            2.0 * (*lapse)() *
                (-2.0 * one_third *
                     (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                     d_extrinsic_curvature(ti_j) +
                 (*inv_conformal_spatial_metric)(ti_K, ti_I) * d_theta(ti_k) +
                 (*conformal_christoffel_second_kind)(ti_I, ti_j, ti_k) *
                     (*inv_a_tilde)(ti_J, ti_K) -
                 3.0 * (*inv_a_tilde)(ti_I, ti_J) * field_p(ti_j) -
                 (*inv_conformal_spatial_metric)(ti_K, ti_I) *
                     (theta() * field_a(ti_k) +
                      2.0 * one_third * trace_extrinsic_curvature() *
                          (*spatial_z4_constraint)(ti_k)) -
                 a_tilde(ti_I, ti_J) * field_a(ti_j) -
                 kappa_1 * (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                     (*spatial_z4_constraint)(ti_j) +
                 // terms with lapse and s
                 (*inv_conformal_spatial_metric)(ti_I, ti_K) *
                     (*inv_conformal_spatial_metric)(ti_N, ti_M) *
                     d_a_tilde(ti_k, ti_n, ti_m) -
                 2.0 * (*inv_conformal_spatial_metric)(ti_I, ti_K) *
                     (*field_d_up)(ti_k, ti_N, ti_M) * a_tilde(ti_n, ti_m)) +
            // terms with s but not not lapse
            (*inv_conformal_spatial_metric)(ti_K, ti_L) *
                (*symmetrized_d_field_b)(ti_k, ti_L, ti_I) +
            one_third * (*inv_conformal_spatial_metric)(ti_I, ti_K) *
                (*contracted_symmetrized_d_field_b)(ti_k));
  }

  // eq. (12i) : time derivative b^i
  // TODO : have a temporary for terms without s? or implement with s known at
  // compile time so that we can if constexpr and only compile one or the other?
  if (s == 0.0) {
    // TODO (?) : add support for assigning to double?
    for (auto& component : *dt_b) {
      component = 0.0;
    }
  } else {
    ::TensorExpressions::evaluate<ti_I>(
        dt_b, shift(ti_K) * (d_b(ti_k, ti_I) - (*d_gamma_hat)(ti_k, ti_I)) +
                  (*dt_gamma_hat)(ti_I)-eta * b(ti_I));
  }

  // eq. (12j) : time derivative of auxiliary variable A_i
  // TODO : extra computatopns with (*lapse)() * (*lapse)()
  // *(d_slicing_condition)() ?
  if (s == 0.0) {
    ::TensorExpressions::evaluate<ti_I>(
        dt_field_a,
        shift(ti_L) * d_field_a(ti_l, ti_k) -
            (*lapse_times_field_a)(ti_k) * (*k_minus_k0_minus_2_theta_c)() *
                (slicing_condition() + (*lapse)() * (d_slicing_condition)()) +
            field_b(ti_k, ti_L) * field_a(ti_l) -
            (*lapse_times_slicing_condition)() *
                (d_trace_extrinsic_curvature(ti_k) - d_k_0(ti_k) -
                 2.0 * c * d_theta(ti_k)));
  } else {
    ::TensorExpressions::evaluate<ti_I>(
        dt_field_a,
        shift(ti_L) * d_field_a(ti_l, ti_k) -
            (*lapse_times_field_a)(ti_k) * (*k_minus_k0_minus_2_theta_c)() *
                (slicing_condition() + (*lapse)() * (d_slicing_condition)()) +
            field_b(ti_k, ti_L) * field_a(ti_l) +
            // terms with \alpha g(\alpha)
            (*lapse_times_slicing_condition)() *
                ((*inv_conformal_metric_times_d_a_tilde)(
                     ti_k)-d_trace_extrinsic_curvature(ti_k) +
                 d_k_0(ti_k) + 2.0 * c * d_theta(ti_k) +
                 2.0 * (*field_d_up_times_a_tilde)(ti_k)));
  }

  // eq. (12k) : time derivative of auxiliary variable B_k{}^i
  if (s == 0.0) {
    // TODO (?) : add support for assigning to double?
    for (auto& component : *dt_b) {
      component = 0.0;
    }
  } else {
    // TODO : extra computations for shift(ti_L) * d_field_b(ti_l, ti_k, ti_I)
    // and square((*lapse)())
    ::TensorExpressions::evaluate<ti_k, ti_I>(
        dt_field_b, shift(ti_L) * d_field_b(ti_l, ti_k, ti_I) +
                        f * d_b(ti_k, ti_I) +
                        mu * square((*lapse)()) *
                            (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                            (d_field_p(ti_k, ti_j) - d_field_p(ti_j, ti_k) -
                             (*inv_conformal_spatial_metric)(ti_N, ti_L) *
                                 (d_field_d(ti_k, ti_l, ti_j, ti_n) -
                                  d_field_d(ti_l, ti_k, ti_j, ti_n))) +
                        field_b(ti_k, ti_L) * field_b(ti_l, ti_I));
  }

  // eq. (12l) : time derivative of auxiliary variable D_{kij}
  if (s == 0.0) {
    ::TensorExpressions::evaluate<ti_k, ti_i, ti_j>(
        dt_field_d,
        shift(ti_L) * d_field_d(ti_l, ti_k, ti_i, ti_j) -
            (*lapse_times_d_a_tilde)(ti_k, ti_i, ti_j) +
            field_b(ti_k, ti_L) * field_d(ti_l, ti_i, ti_j) +
            (*field_d_times_field_b)(ti_j, ti_k, ti_i) +
            (*field_d_times_field_b)(ti_i, ti_k, ti_j) -
            (*lapse_times_field_a)(ti_k) *
                (*a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde)(
                    ti_i, ti_j) +
            one_third *
                ((*lapse_times_conformal_spatial_metric)(ti_i, ti_j) *
                     (*inv_conformal_metric_times_d_a_tilde)(ti_k)-2.0 *
                     (*contracted_field_b)() * field_b(ti_k, ti_i, ti_j) -
                 2.0 * (*lapse_times_conformal_spatial_metric)(ti_i, ti_j) *
                     (*field_d_up_times_a_tilde)(ti_k)));
  } else {
    ::TensorExpressions::evaluate<ti_k, ti_i, ti_j>(
        dt_field_d,
        shift(ti_L) * d_field_d(ti_l, ti_k, ti_i, ti_j) +
            0.25 * ((*conformal_metric_times_field_b)(ti_i, ti_k, ti_j) +
                    (*conformal_metric_times_field_b)(ti_i, ti_j, ti_k) +
                    (*conformal_metric_times_field_b)(ti_j, ti_k, ti_i) +
                    (*conformal_metric_times_field_b)(ti_j, ti_i, ti_k)) -
            (*lapse_times_d_a_tilde)(ti_k, ti_i, ti_j) +
            field_b(ti_k, ti_L) * field_d(ti_l, ti_i, ti_j) +
            (*field_d_times_field_b)(ti_j, ti_k, ti_i) +
            (*field_d_times_field_b)(ti_i, ti_k, ti_j) -
            (*lapse_times_field_a)(ti_k) *
                (*a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde)(
                    ti_i, ti_j) +
            one_third *
                ((*lapse_times_conformal_spatial_metric)(ti_i, ti_j) *
                     (*inv_conformal_metric_times_d_a_tilde)(ti_k)-2.0 *
                     (*contracted_field_b)() * field_b(ti_k, ti_i, ti_j) -
                 2.0 * (*lapse_times_conformal_spatial_metric)(ti_i, ti_j) *
                     (*field_d_up_times_a_tilde)(ti_k)-conformal_spatial_metric(
                         ti_i, ti_j) *
                     (*contracted_symmetrized_d_field_b)(ti_k)));
  }

  // eq. (12m) : time derivative of auxiliary variable P_i
  // TODO
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data) template struct Ccz4::TimeDerivative<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
