// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
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
  using pi_one_normal_type = tnsr::a<DataType, Dim>;
  using inverse_spatial_metric_type = tnsr::II<DataType, Dim>;
  using d_phi_type = tnsr::ijaa<DataType, Dim>;
  using lapse_type = Scalar<DataType>;
  using gamma1gamma2_type = Scalar<DataType>;
  using shift_dot_three_index_constraint_type = tnsr::aa<DataType, Dim>;
  using shift_type = tnsr::I<DataType, Dim>;
  using d_pi_type = tnsr::iaa<DataType, Dim>;

  SPECTRE_ALWAYS_INLINE static void apply(
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
      const pi_one_normal_type& pi_one_normal,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const d_phi_type& d_phi, const lapse_type& lapse,
      const gamma1gamma2_type& gamma1gamma2,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const d_pi_type& d_pi) {
    destructive_resize_components(dt_pi, get_size(get(gamma0)));

    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        dt_pi->get(mu, nu) =
            -spacetime_deriv_gauge_function.get(mu, nu) -
            spacetime_deriv_gauge_function.get(nu, mu) -
            0.5 * get(pi_two_normals) * pi.get(mu, nu) +
            get(gamma0) *
                (normal_spacetime_one_form.get(mu) * gauge_constraint.get(nu) +
                 normal_spacetime_one_form.get(nu) * gauge_constraint.get(mu)) -
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
              pi_one_normal.get(m + 1) * phi_1_up.get(m, mu, nu);

          for (size_t n = 0; n < Dim; ++n) {
            dt_pi->get(mu, nu) -=
                inverse_spatial_metric.get(m, n) * d_phi.get(m, n, mu, nu);
          }
        }

        dt_pi->get(mu, nu) *= get(lapse);

        dt_pi->get(mu, nu) +=
            get(gamma1gamma2) * shift_dot_three_index_constraint.get(mu, nu);

        for (size_t m = 0; m < Dim; ++m) {
          // DualFrame term
          dt_pi->get(mu, nu) += shift.get(m) * d_pi.get(m, mu, nu);
        }
      }
    }
  }
};
