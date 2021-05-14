// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
  // tensor types in tensor equation being benchmarked
  using dt_pi_type = tnsr::aa<DataType, Dim>;
  using spacetime_deriv_gauge_function_type = tnsr::ab<DataType, Dim>;
  using pi_two_normals_type = Scalar<DataType>;
  using pi_type = tnsr::aa<DataType, Dim>;
  using gamma0_type = Scalar<DataType>;
  using normal_spacetime_one_form_type = tnsr::a<DataType, Dim>;
  using gauge_constraint_type = tnsr::a<DataType, Dim>;
  using spacetime_metric_type = tnsr::aa<DataType, Dim>;
  using normal_dot_gauge_constraint_type = Scalar<DataType>;
  using christoffel_second_kind_type = tnsr::Abb<DataType, Dim>;
  using gauge_function_type = tnsr::a<DataType, Dim>;
  using pi_2_up_type = tnsr::aB<DataType, Dim>;
  using phi_1_up_type = tnsr::Iaa<DataType, Dim>;
  using phi_3_up_type = tnsr::iaB<DataType, Dim>;
  using christoffel_first_kind_3_up_type = tnsr::abC<DataType, Dim>;
  // type not in SpECTRE implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type = tnsr::i<DataType, Dim>;
  using inverse_spatial_metric_type = tnsr::II<DataType, Dim>;
  using d_phi_type = tnsr::ijaa<DataType, Dim>;
  using lapse_type = Scalar<DataType>;
  using gamma1gamma2_type = Scalar<DataType>;
  using shift_dot_three_index_constraint_type = tnsr::aa<DataType, Dim>;
  using shift_type = tnsr::I<DataType, Dim>;
  using d_pi_type = tnsr::iaa<DataType, Dim>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<dt_pi_type*> dt_pi,
      const spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function,
      const pi_two_normals_type& pi_two_normals, const pi_type& pi,
      const gamma0_type& gamma0,
      const normal_spacetime_one_form_type& normal_spacetime_one_form,
      const gauge_constraint_type& gauge_constraint,
      const spacetime_metric_type& spacetime_metric,
      const normal_dot_gauge_constraint_type& normal_dot_gauge_constraint,
      const christoffel_second_kind_type& christoffel_second_kind,
      const gauge_function_type& gauge_function, const pi_2_up_type& pi_2_up,
      const phi_1_up_type& phi_1_up, const phi_3_up_type& phi_3_up,
      const christoffel_first_kind_3_up_type& christoffel_first_kind_3_up,
      const pi_one_normal_spatial_type& pi_one_normal_spatial,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const d_phi_type& d_phi, const lapse_type& lapse,
      const gamma1gamma2_type& gamma1gamma2,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const d_pi_type& d_pi) noexcept {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        dt_pi->get(mu, nu) =
            -spacetime_deriv_gauge_function.get(mu, nu) -
            spacetime_deriv_gauge_function.get(nu, mu) -
            0.5 * get(pi_two_normals) * pi.get(mu, nu) +
            get(gamma0) * normal_spacetime_one_form.get(mu) *
                gauge_constraint.get(nu) +
            normal_spacetime_one_form.get(nu) * gauge_constraint.get(mu) -
            get(gamma0) * spacetime_metric.get(mu, nu) *
                get(normal_dot_gauge_constraint);

        for (size_t delta = 0; delta < Dim + 1; ++delta) {
          dt_pi->get(mu, nu) += 2 * christoffel_second_kind.get(delta, mu, nu) *
                                    gauge_function.get(delta) -
                                2 * pi.get(mu, delta) * pi_2_up.get(nu, delta);

          for (size_t n = 0; n < Dim; ++n) {
            dt_pi->get(mu, nu) +=
                2 * phi_1_up.get(n, mu, delta) * phi_3_up.get(n, nu, delta);
          }

          for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
            dt_pi->get(mu, nu) -=
                2. * christoffel_first_kind_3_up.get(mu, alpha, delta) *
                christoffel_first_kind_3_up.get(nu, delta, alpha);
          }
        }

        for (size_t m = 0; m < Dim; ++m) {
          dt_pi->get(mu, nu) -=
              pi_one_normal_spatial.get(m) * phi_1_up.get(m, mu, nu);

          for (size_t n = 0; n < Dim; ++n) {
            dt_pi->get(mu, nu) -=
                inverse_spatial_metric.get(m, n) * d_phi.get(m, n, mu, nu);
          }
        }

        dt_pi->get(mu, nu) *= get(lapse);

        dt_pi->get(mu, nu) +=
            get(gamma1gamma2) * shift_dot_three_index_constraint.get(mu, nu);

        for (size_t m = 0; m < Dim; ++m) {
          dt_pi->get(mu, nu) += shift.get(m) * d_pi.get(m, mu, nu);
        }
      }
    }
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<dt_pi_type*> dt_pi,
      const spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function,
      const pi_two_normals_type& pi_two_normals, const pi_type& pi,
      const gamma0_type& gamma0,
      const normal_spacetime_one_form_type& normal_spacetime_one_form,
      const gauge_constraint_type& gauge_constraint,
      const spacetime_metric_type& spacetime_metric,
      const normal_dot_gauge_constraint_type& normal_dot_gauge_constraint,
      const christoffel_second_kind_type& christoffel_second_kind,
      const gauge_function_type& gauge_function, const pi_2_up_type& pi_2_up,
      const phi_1_up_type& phi_1_up, const phi_3_up_type& phi_3_up,
      const christoffel_first_kind_3_up_type& christoffel_first_kind_3_up,
      const pi_one_normal_spatial_type& pi_one_normal_spatial,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const d_phi_type& d_phi, const lapse_type& lapse,
      const gamma1gamma2_type& gamma1gamma2,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const d_pi_type& d_pi) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<dt_pi_type*> dt_pi,
      const spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function,
      const pi_two_normals_type& pi_two_normals, const pi_type& pi,
      const gamma0_type& gamma0,
      const normal_spacetime_one_form_type& normal_spacetime_one_form,
      const gauge_constraint_type& gauge_constraint,
      const spacetime_metric_type& spacetime_metric,
      const normal_dot_gauge_constraint_type& normal_dot_gauge_constraint,
      const christoffel_second_kind_type& christoffel_second_kind,
      const gauge_function_type& gauge_function, const pi_2_up_type& pi_2_up,
      const phi_1_up_type& phi_1_up, const phi_3_up_type& phi_3_up,
      const christoffel_first_kind_3_up_type& christoffel_first_kind_3_up,
      const pi_one_normal_spatial_type& pi_one_normal_spatial,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const d_phi_type& d_phi, const lapse_type& lapse,
      const gamma1gamma2_type& gamma1gamma2,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const d_pi_type& d_pi) noexcept {
    TensorExpressions::evaluate<ti_a, ti_b>(
        dt_pi,
        ((-1.0 * spacetime_deriv_gauge_function(ti_a, ti_b)) -
         spacetime_deriv_gauge_function(ti_b, ti_a) -
         0.5 * pi_two_normals() * pi(ti_a, ti_b) +
         gamma0() * (normal_spacetime_one_form(ti_a) * gauge_constraint(ti_b) +
                     normal_spacetime_one_form(ti_b) * gauge_constraint(ti_a)) -
         gamma0() * spacetime_metric(ti_a, ti_b) *
             normal_dot_gauge_constraint() +
         2.0 * christoffel_second_kind(ti_C, ti_a, ti_b) *
             gauge_function(ti_c) -
         2.0 * pi(ti_a, ti_c) * pi_2_up(ti_b, ti_C) +
         2.0 * phi_1_up(ti_I, ti_a, ti_c) * phi_3_up(ti_i, ti_b, ti_C) -
         2.0 * christoffel_first_kind_3_up(ti_a, ti_d, ti_C) *
             christoffel_first_kind_3_up(ti_b, ti_c, ti_D) -
         pi_one_normal_spatial(ti_j) * phi_1_up(ti_J, ti_a, ti_b) -
         inverse_spatial_metric(ti_J, ti_K) * d_phi(ti_j, ti_k, ti_a, ti_b)) *
                lapse() +
            gamma1gamma2() * shift_dot_three_index_constraint(ti_a, ti_b) +
            shift(ti_I) * d_pi(ti_i, ti_a, ti_b));
  }
};
