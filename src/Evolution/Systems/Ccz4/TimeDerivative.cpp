// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CCZ4/TimeDerivative.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace CCZ4 {
template <size_t Dim>
void TimeDerivative<Dim>::apply(
    // time derivatives
    const gsl::not_null<tnsr::ij<DataVector, Dim>*> dt_conf_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> dt_ln_lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_shift,
    const gsl::not_null<Scalar<DataVector>*> dt_ln_phi,
    // evolved variables
    const gsl::not_null<Scalar<DataVector>*> trace_a_tilde,
    // temporary expressions
    const gsl::not_null<Scalar<DataVector>*> k_minus_2_theta_c,
    const gsl::not_null<Scalar<DataVector>*> k_minus_k0_minus_2_theta_c,
    const gsl::not_null<Scalar<DataVector>*> contracted_field_b,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_ricci_scalar_plus_2_div_z4_constraint,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        conf_metric_times_trace_a_tilde,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*> lapse_times_a_tilde,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> field_d_up_times_a_tilde,
    const gsl::not_null<tnsr::ijj<DataVector, Dim>*> lapse_times_d_a_tilde,
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        inv_conf_metric_times_d_a_tilde,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        a_tilde_minus_one_third_conf_metric_times_trace_a_tilde,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> lapse_times_field_a,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> shift_times_deriv_gamma_hat,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*> inv_tau_times_conf_metric,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_slicing_condition,
    // auxiliary variables
    const tnsr::i<DataVector, Dim>& field_a,
    const tnsr::iJ<DataVector, Dim>& field_b,
    const tnsr::ijk<DataVector, Dim>& field_d,
    const tnsr::i<DataVector, Dim>& field_p,
    // other (TODO: sort)
    const tnsr::ii<DataVector, Dim>& conf_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> det_conf_spatial_metric,
    const tnsr::II<DataVector, Dim>& inv_conf_spatial_metric,
    const gsl::not_null<tnsr::iJJ<DataVector, Dim>*> field_d_up,
    const tnsr::I<DataVector, Dim>& shift,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::I<DataVector, Dim>& gamma_hat,
    const tnsr::iJ<DataVector, Dim>& d_gamma_hat, const Scalar<DataVector>& k_0,
    const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& slicing_condition, const Scalar<DataVector>& g,
    const Scalar<DataVector>& theta, const double c,
    const tnsr::ii<DataVector, Dim>& a_tilde,
    const tnsr::ijj<DataVector, Dim>& d_a_tilde,
    const double one_over_relaxation_time, const double s, const double f,
    const tnsr::I<DataVector, Dim>& b,
    const Scalar<DataVector>& ricci_scalar_plus_2_div_z4_constraint) {
  // Repeated temporaries
  TensorExpressions::evaluate(k_minus_2_theta_c, 2.0 * theta() * c);

  TensorExpressions::evaluate(k_minus_k0_minus_2_theta_c,
                              (*k_minus_2_theta_c)() - k_0());

  TensorExpressions::evaluate(contracted_field_b, field_b(ti_k, ti_K));

  //   TensorExpressions::evaluate(
  //       lapse_times_ricci_scalar_plus_2_div_z4_constraint,
  //       lapse() * ricci_scalar_plus_2_div_z4_constraint());

  TensorExpressions::evaluate<ti_i, ti_j>(
      conf_metric_times_trace_a_tilde,
      conf_spatial_metric(ti_i, ti_j) * (*trace_a_tilde()));

  //   TensorExpressions::evaluate<ti_i, ti_j>(lapse_times_a_tilde,
  //                                           lapse() * a_tilde(ti_i, ti_j));

  //   TensorExpressions::evaluate<ti_k>(
  //       field_d_up_times_a_tilde,
  //       (*field_d_up)(ti_k, ti_I, ti_J) * a_tilde(ti_i, ti_j));

  //   // TODO: how to get deriv of \tilde{A}_{ij}?
  //   TensorExpressions::evaluate<ti_k, ti_i, ti_j>(
  //       lapse_times_d_a_tilde, lapse() * d_a_tilde(ti_k, ti_i, ti_j));

  //   TensorExpressions::evaluate<ti_k>(
  //       inv_conf_metric_times_d_a_tilde,
  //       inv_conf_spatial_metric(ti_I, ti_J) * d_a_tilde(ti_k, ti_i, ti_j));

  TensorExpressions::evaluate<ti_i, ti_j>(
      a_tilde_minus_one_third_conf_metric_times_trace_a_tilde,
      a_tilde(ti_i, ti_j) -
          (1.0 / 3) * conf_metric_times_trace_a_tilde(ti_i, ti_j));

  //   TensorExpressions::evaluate<ti_k>(lapse_times_field_a,
  //                                     lapse() * filed_a(ti_k));

  //   TensorExpressions::evaluate<ti_I>(shift_times_deriv_gamma_hat,
  //                                     shift(ti_K) * d_gamma_hat(ti_k, ti_I));

  TensorExpressions::evaluate<ti_i, ti_j>(
      inv_tau_times_conf_metric,
      one_over_relaxation_time * conf_spatial_metric(ti_i, ti_j));

  // TODO: do we actually want this? this quantity will be = 1 or = lapse
  //   TensorExpressions::evaluate(lapse_times_slicing_condition,
  //                               lapse() * slicing_condition());

  // TODO: add this to ccz4-temp-tags PR
  TensorExpressions::evaluate<ti_i, ti_j>(
      conf_metric_times_field_b,
      conf_spatial_metric(ti_k, ti_i) * field_b(ti_j, ti_K));

  // Time derivative computation

  // dt_conf_spatial_metric: time derivative of the conformal spatial metric
  ::TensorExpressions::evaluate<ti_i, ti_j>(
      dt_conf_spatial_metric,
      2.0 * shift(ti_K) * field_d(ti_k, ti_i, ti_j) +
          (*conf_metric_times_field_b)(ti_i, ti_j) +
          (*conf_metric_times_field_b)(ti_j, ti_i) -
          (2.0 / 3) * conf_spatial_metric(ti_i, ti_j) *
              (*contracted_field_b)() -
          2.0 * lapse() *
              (a_tilde_minus_one_third_conf_metric_times_trace_a_tilde)(ti_i,
                                                                        ti_j) -
          (*inv_tau_times_conf_metric)(ti_i, ti_j) *
              ((*det_conf_spatial_metric)() - 1));

  // dt_ln_lapse: time derivative of the natural log of the lapse
  ::TensorExpressions::evaluate(
      dt_ln_lapse,
      shift(ti_K) * field_a(ti_k) -
          lapse() * slicing_condition() * (*k_minus_k0_minus_2_theta_c)());

  // dt_shift: time derivative of the shift
  ::TensorExpressions::evaluate<ti_I>(
      dt_shift, s * shift(ti_K) * B(ti_k, ti_I) + s * f * b(ti_I));

  // dt_ln_phi: time derivative of the natural log of the conformal factor
  ::TensorExpressions::evaluate(
      dt_ln_phi, shift(ti_K) * field_P(ti_k) +
                     (1.0 / 3) * (lapse() * trace_extrinsic_curvature() -
                                  (*contracted_field_b)()));
}
}  // namespace CCZ4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data) template struct CCZ4::TimeDerivative<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
