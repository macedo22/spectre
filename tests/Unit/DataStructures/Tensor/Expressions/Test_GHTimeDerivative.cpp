// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>  // for std::uniform_real_distribution
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {

constexpr size_t Dim = 3;
constexpr size_t num_grid_points = 5;

using dt_spacetime_metric_type = tnsr::aa<DataVector, Dim>;
using dt_pi_type = tnsr::aa<DataVector, Dim>;
using dt_phi_type = tnsr::iaa<DataVector, Dim>;
using temp_gamma1_type = Scalar<DataVector>;
using temp_gamma2_type = Scalar<DataVector>;
using gamma1gamma2_type = Scalar<DataVector>;
using pi_two_normals_type = Scalar<DataVector>;
using normal_dot_gauge_constraint_type = Scalar<DataVector>;
using gamma1_plus_1_type = Scalar<DataVector>;
using pi_one_normal_type = tnsr::a<DataVector, Dim>;
using gauge_constraint_type = tnsr::a<DataVector, Dim>;
using phi_two_normals_type = tnsr::i<DataVector, Dim>;
using shift_dot_three_index_constraint_type = tnsr::aa<DataVector, Dim>;
using phi_one_normal_type = tnsr::ia<DataVector, Dim>;
using pi_2_up_type = tnsr::aB<DataVector, Dim>;
using three_index_constraint_type = tnsr::iaa<DataVector, Dim>;
using phi_1_up_type = tnsr::Iaa<DataVector, Dim>;
using phi_3_up_type = tnsr::iaB<DataVector, Dim>;
using christoffel_first_kind_3_up_type = tnsr::abC<DataVector, Dim>;
using lapse_type = Scalar<DataVector>;
using shift_type = tnsr::I<DataVector, Dim>;
using spatial_metric_type = tnsr::ii<DataVector, Dim>;
using inverse_spatial_metric_type = tnsr::II<DataVector, Dim>;
using det_spatial_metric_type = Scalar<DataVector>;
using inverse_spacetime_metric_type = tnsr::AA<DataVector, Dim>;
using christoffel_first_kind_type = tnsr::abb<DataVector, Dim>;
using christoffel_second_kind_type = tnsr::Abb<DataVector, Dim>;
using trace_christoffel_type = tnsr::a<DataVector, Dim>;
using normal_spacetime_vector_type = tnsr::A<DataVector, Dim>;
using normal_spacetime_one_form_type = tnsr::a<DataVector, Dim>;
using da_spacetime_metric_type = tnsr::abb<DataVector, Dim>;
using d_spacetime_metric_type = tnsr::iaa<DataVector, Dim>;
using d_pi_type = tnsr::iaa<DataVector, Dim>;
using d_phi_type = tnsr::ijaa<DataVector, Dim>;
using spacetime_metric_type = tnsr::aa<DataVector, Dim>;
using pi_type = tnsr::aa<DataVector, Dim>;
using phi_type = tnsr::iaa<DataVector, Dim>;
using gamma0_type = Scalar<DataVector>;
using gamma1_type = Scalar<DataVector>;
using gamma2_type = Scalar<DataVector>;
using gauge_function_type = tnsr::a<DataVector, Dim>;
using spacetime_deriv_gauge_function_type = tnsr::ab<DataVector, Dim>;

void compute_expected_result_impl(
    const gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_spacetime_metric,
    const gsl::not_null<dt_pi_type*> dt_pi,
    const gsl::not_null<dt_phi_type*> dt_phi,
    const gsl::not_null<temp_gamma1_type*> temp_gamma1,
    const gsl::not_null<temp_gamma2_type*> temp_gamma2,
    const gsl::not_null<gamma1gamma2_type*> gamma1gamma2,
    const gsl::not_null<pi_two_normals_type*> pi_two_normals,
    const gsl::not_null<normal_dot_gauge_constraint_type*>
        normal_dot_gauge_constraint,
    const gsl::not_null<gamma1_plus_1_type*> gamma1_plus_1,
    const gsl::not_null<pi_one_normal_type*> pi_one_normal,
    const gsl::not_null<gauge_constraint_type*> gauge_constraint,
    const gsl::not_null<phi_two_normals_type*> phi_two_normals,
    const gsl::not_null<shift_dot_three_index_constraint_type*>
        shift_dot_three_index_constraint,
    const gsl::not_null<phi_one_normal_type*> phi_one_normal,
    const gsl::not_null<pi_2_up_type*> pi_2_up,
    const gsl::not_null<three_index_constraint_type*> three_index_constraint,
    const gsl::not_null<phi_1_up_type*> phi_1_up,
    const gsl::not_null<phi_3_up_type*> phi_3_up,
    const gsl::not_null<christoffel_first_kind_3_up_type*>
        christoffel_first_kind_3_up,
    const gsl::not_null<lapse_type*> lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> shift,
    const gsl::not_null<spatial_metric_type*> spatial_metric,
    const gsl::not_null<inverse_spatial_metric_type*> inverse_spatial_metric,
    const gsl::not_null<det_spatial_metric_type*> det_spatial_metric,
    const gsl::not_null<inverse_spacetime_metric_type*>
        inverse_spacetime_metric,
    const gsl::not_null<christoffel_first_kind_type*> christoffel_first_kind,
    const gsl::not_null<christoffel_second_kind_type*> christoffel_second_kind,
    const gsl::not_null<trace_christoffel_type*> trace_christoffel,
    const gsl::not_null<normal_spacetime_vector_type*> normal_spacetime_vector,
    const gsl::not_null<normal_spacetime_one_form_type*>
        normal_spacetime_one_form,
    const gsl::not_null<da_spacetime_metric_type*> da_spacetime_metric,
    const d_spacetime_metric_type& d_spacetime_metric, const d_pi_type& d_pi,
    const d_phi_type& d_phi, const spacetime_metric_type& spacetime_metric,
    const pi_type& pi, const phi_type& phi, const gamma0_type& gamma0,
    const gamma1_type& gamma1, const gamma2_type& gamma2,
    const gauge_function_type& gauge_function,
    const spacetime_deriv_gauge_function_type&
        spacetime_deriv_gauge_function) noexcept {
  // Need constraint damping on interfaces in DG schemes
  *temp_gamma1 = gamma1;
  *temp_gamma2 = gamma2;

  // can't do with TE's yet
  gr::spatial_metric(spatial_metric, spacetime_metric);
  // can't do with TE's yet
  determinant_and_inverse(det_spatial_metric, inverse_spatial_metric,
                          *spatial_metric);
  // can't do with TE's yet
  gr::shift(shift, spacetime_metric, *inverse_spatial_metric);
  // can't do with TE's yet
  gr::lapse(lapse, *shift, spacetime_metric);
  // can't do with TE's yet
  gr::inverse_spacetime_metric(inverse_spacetime_metric, *lapse, *shift,
                               *inverse_spatial_metric);
  // can't do with TE's yet
  GeneralizedHarmonic::spacetime_derivative_of_spacetime_metric(
      da_spacetime_metric, *lapse, *shift, pi, phi);
  // Note: 2nd and 3rd indices of result are symmetric
  // auto christoffel_first_kind = evaluate<ti_c, ti_a, ti_b>(
  //    0.5 * (d_metric(ti_a, ti_b, ti_c) + d_metric(ti_b, ti_a, ti_c) -
  //           d_metric(ti_c, ti_a, ti_b)));
  gr::christoffel_first_kind(christoffel_first_kind, *da_spacetime_metric);
  // Note: 2nd and 3rd indices of result are symmetric
  // auto christoffel_second_kind = evaluate<ti_A, ti_b, ti_c>(
  //    christoffel_first_kind(ti_d, ti_b, ti_c) *
  //    inverse_spacetime_metric(ti_A, ti_D));
  raise_or_lower_first_index(christoffel_second_kind, *christoffel_first_kind,
                             *inverse_spacetime_metric);
  // auto trace_christoffel = evaluate<ti_a>(
  //    christoffel_first_kind(ti_a, ti_b, ti_c) *
  //    inverse_spacetime_metric(ti_B, ti_C));
  trace_last_indices(trace_christoffel, *christoffel_first_kind,
                     *inverse_spacetime_metric);
  // can't do with TE's yet
  gr::spacetime_normal_vector(normal_spacetime_vector, *lapse, *shift);
  // can't do with TE's yet
  gr::spacetime_normal_one_form(normal_spacetime_one_form, *lapse);

  // auto gamma1gamma2 = evaluate(gamma1 * gamma 2);
  get(*gamma1gamma2) = get(gamma1) * get(gamma2);
  // not an eq
  const DataVector& gamma12 = get(*gamma1gamma2);

  // Note: 2nd and 3rd indices of result are symmetric
  // auto phi_1_up = evaluate<ti_I, ti_a, ti_b>(
  //     inverse_spatial_metric(ti_I, ti_J) * // II
  //     phi(ti_j, ti_a, ti_b));              // iaa
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        phi_1_up->get(m, mu, nu) =
            inverse_spatial_metric->get(m, 0) * phi.get(0, mu, nu);
        for (size_t n = 1; n < Dim; ++n) {
          phi_1_up->get(m, mu, nu) +=
              inverse_spatial_metric->get(m, n) * phi.get(n, mu, nu);
        }
      }
    }
  }

  // auto phi_3_up = evaluate<ti_i, ti_a, ti_B>( // iaB
  //     inverse_spacetime_metric(ti_B, ti_C) *  // AA
  //     phi(ti_i, ti_a, ti_c));                 // iaa
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        phi_3_up->get(m, nu, alpha) =
            inverse_spacetime_metric->get(alpha, 0) * phi.get(m, nu, 0);
        for (size_t beta = 1; beta < Dim + 1; ++beta) {
          phi_3_up->get(m, nu, alpha) +=
              inverse_spacetime_metric->get(alpha, beta) * phi.get(m, nu, beta);
        }
      }
    }
  }

  // auto pi_2_up = evaluate<ti_a, ti_B>(        // aB
  //     inverse_spacetime_metric(ti_B, ti_C) *  // AA
  //     pi(ti_a, ti_c));                        // aa
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
      pi_2_up->get(nu, alpha) =
          inverse_spacetime_metric->get(alpha, 0) * pi.get(nu, 0);
      for (size_t beta = 1; beta < Dim + 1; ++beta) {
        pi_2_up->get(nu, alpha) +=
            inverse_spacetime_metric->get(alpha, beta) * pi.get(nu, beta);
      }
    }
  }

  // auto christoffel_first_kind_3_up = evaluate<ti_a, ti_b, ti_C>(  //abC
  //     inverse_spacetime_metric(ti_C, ti_D) *                  // AA
  //     christoffel_first_kind(ti_a, ti_b, ti_d));              // abb
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        christoffel_first_kind_3_up->get(mu, nu, alpha) =
            inverse_spacetime_metric->get(alpha, 0) *
            christoffel_first_kind->get(mu, nu, 0);
        for (size_t beta = 1; beta < Dim + 1; ++beta) {
          christoffel_first_kind_3_up->get(mu, nu, alpha) +=
              inverse_spacetime_metric->get(alpha, beta) *
              christoffel_first_kind->get(mu, nu, beta);
        }
      }
    }
  }

  // auto pi_one_normal = evaluate<ti_a>(  // a
  //     normal_spacetime_vector(ti_B) *   // A
  //     pi(ti_b, ti_a));                  // aa
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    pi_one_normal->get(mu) = get<0>(*normal_spacetime_vector) * pi.get(0, mu);
    for (size_t nu = 1; nu < Dim + 1; ++nu) {
      pi_one_normal->get(mu) +=
          normal_spacetime_vector->get(nu) * pi.get(nu, mu);
    }
  }

  // auto pi_two_normals = evaluate(       // scalar
  //     normal_spacetime_vector(ti_A) *   // A
  //     pi_one_normal(ti_a));             // a
  get(*pi_two_normals) =
      get<0>(*normal_spacetime_vector) * get<0>(*pi_one_normal);
  for (size_t mu = 1; mu < Dim + 1; ++mu) {
    get(*pi_two_normals) +=
        normal_spacetime_vector->get(mu) * pi_one_normal->get(mu);
  }

  // auto phi_one_normal = evaluate<ti_i, ti_a>(  // ia
  //     normal_spacetime_vector(ti_B) *          // A
  //     phi(ti_i, ti_b, ti_a));                  // iaa
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      phi_one_normal->get(n, nu) =
          get<0>(*normal_spacetime_vector) * phi.get(n, 0, nu);
      for (size_t mu = 1; mu < Dim + 1; ++mu) {
        phi_one_normal->get(n, nu) +=
            normal_spacetime_vector->get(mu) * phi.get(n, mu, nu);
      }
    }
  }

  // auto phi_two_normals = evaluate<ti_i, ti_a>( // i
  //     normal_spacetime_vector(ti_A) *          // A
  //     phi_one_normal(ti_i, ti_a));             // ia
  for (size_t n = 0; n < Dim; ++n) {
    phi_two_normals->get(n) =
        get<0>(*normal_spacetime_vector) * phi_one_normal->get(n, 0);
    for (size_t mu = 1; mu < Dim + 1; ++mu) {
      phi_two_normals->get(n) +=
          normal_spacetime_vector->get(mu) * phi_one_normal->get(n, mu);
    }
  }

  // Note: 2nd and 3rd indices of result are symmetric
  // auto three_index_constraint = evaluate<ti_i, ti_a, ti_b>( // iaa
  //     d_spacetime_metric(ti_i, ti_a, ti_b) -                // iaa
  //     phi(ti_i, ti_a, ti_b));                               // iaa
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        three_index_constraint->get(n, mu, nu) =
            d_spacetime_metric.get(n, mu, nu) - phi.get(n, mu, nu);
      }
    }
  }

  // auto gauge_constraint = evaluate<ti_a>( // a
  //     gauge_function(ti_a) +              // a
  //     trace_christoffel(ti_a));           // a
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    gauge_constraint->get(nu) =
        gauge_function.get(nu) + trace_christoffel->get(nu);
  }

  // auto gamma1_plus_1 = evaluate(1.0 + gamma1()); // scalar
  get(*gamma1_plus_1) = 1.0 + gamma1.get();
  // not an eq
  const DataVector& gamma1p1 = get(*gamma1_plus_1);

  // auto shift_dot_three_index_constraint = evaluate<ti_a, ti_b>( // aa
  //     shift(ti_I) +                                             // I
  //     three_index_constraint(ti_i, ti_a, ti_b));                // iaa
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      shift_dot_three_index_constraint->get(mu, nu) =
          get<0>(*shift) * three_index_constraint->get(0, mu, nu);
      for (size_t m = 1; m < Dim; ++m) {
        shift_dot_three_index_constraint->get(mu, nu) +=
            shift->get(m) * three_index_constraint->get(m, mu, nu);
      }
    }
  }

  // Equation for dt_spacetime_metric
  // dt_spacetime_metric : aa
  // lapse: scalar
  // pi: aa
  // gamma1p1 (gamma1_plus_1) : scalar
  // shift_dot_three_index_constraint : aa
  // shift : I
  // phi : iaa
  // auto dt_spacetime_metric = evaluate<ti_a, ti_b> (
  //    -1.0 * lapse() * pi(ti_a, ti_b) +
  //     gamma1_plus_1() * shift_dot_three_index_constraint(ti_a, ti_b) +
  //     + shift(ti_I) * phi(ti_i, ti_a, ti_b)));
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      dt_spacetime_metric->get(mu, nu) = -get(*lapse) * pi.get(mu, nu);
      dt_spacetime_metric->get(mu, nu) +=
          gamma1p1 * shift_dot_three_index_constraint->get(mu, nu);
      for (size_t m = 0; m < Dim; ++m) {
        dt_spacetime_metric->get(mu, nu) += shift->get(m) * phi.get(m, mu, nu);
      }
    }
  }
}

void compute_expected_result(
    gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
    const d_spacetime_metric_type& d_spacetime_metric, const d_pi_type& d_pi,
    const d_phi_type& d_phi, const spacetime_metric_type& spacetime_metric,
    const pi_type& pi, const phi_type& phi, const gamma0_type& gamma0,
    const gamma1_type& gamma1, const gamma2_type& gamma2,
    const gauge_function_type& gauge_function,
    const spacetime_deriv_gauge_function_type&
        spacetime_deriv_gauge_function) noexcept {
  // create tensors to be filled by implementation
  dt_pi_type dt_pi(num_grid_points);
  dt_phi_type dt_phi(num_grid_points);
  temp_gamma1_type temp_gamma1(num_grid_points);
  temp_gamma2_type temp_gamma2(num_grid_points);
  gamma1gamma2_type gamma1gamma2(num_grid_points);
  pi_two_normals_type pi_two_normals(num_grid_points);
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint(num_grid_points);
  gamma1_plus_1_type gamma1_plus_1(num_grid_points);
  pi_one_normal_type pi_one_normal(num_grid_points);
  gauge_constraint_type gauge_constraint(num_grid_points);
  phi_two_normals_type phi_two_normals(num_grid_points);
  shift_dot_three_index_constraint_type shift_dot_three_index_constraint(
      num_grid_points);
  phi_one_normal_type phi_one_normal(num_grid_points);
  pi_2_up_type pi_2_up(num_grid_points);
  three_index_constraint_type three_index_constraint(num_grid_points);
  phi_1_up_type phi_1_up(num_grid_points);
  phi_3_up_type phi_3_up(num_grid_points);
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up(num_grid_points);
  lapse_type lapse(num_grid_points);
  shift_type shift(num_grid_points);
  spatial_metric_type spatial_metric(num_grid_points);
  inverse_spatial_metric_type inverse_spatial_metric(num_grid_points);
  det_spatial_metric_type det_spatial_metric(num_grid_points);
  inverse_spacetime_metric_type inverse_spacetime_metric(num_grid_points);
  christoffel_first_kind_type christoffel_first_kind(num_grid_points);
  christoffel_second_kind_type christoffel_second_kind(num_grid_points);
  trace_christoffel_type trace_christoffel(num_grid_points);
  normal_spacetime_vector_type normal_spacetime_vector(num_grid_points);
  normal_spacetime_one_form_type normal_spacetime_one_form(num_grid_points);
  da_spacetime_metric_type da_spacetime_metric(num_grid_points);

  compute_expected_result_impl(
      dt_spacetime_metric, make_not_null(&dt_pi), make_not_null(&dt_phi),
      make_not_null(&temp_gamma1), make_not_null(&temp_gamma2),
      make_not_null(&gamma1gamma2), make_not_null(&pi_two_normals),
      make_not_null(&normal_dot_gauge_constraint),
      make_not_null(&gamma1_plus_1), make_not_null(&pi_one_normal),
      make_not_null(&gauge_constraint), make_not_null(&phi_two_normals),
      make_not_null(&shift_dot_three_index_constraint),
      make_not_null(&phi_one_normal), make_not_null(&pi_2_up),
      make_not_null(&three_index_constraint), make_not_null(&phi_1_up),
      make_not_null(&phi_3_up), make_not_null(&christoffel_first_kind_3_up),
      make_not_null(&lapse), make_not_null(&shift),
      make_not_null(&spatial_metric), make_not_null(&inverse_spatial_metric),
      make_not_null(&det_spatial_metric),
      make_not_null(&inverse_spacetime_metric),
      make_not_null(&christoffel_first_kind),
      make_not_null(&christoffel_second_kind),
      make_not_null(&trace_christoffel),
      make_not_null(&normal_spacetime_vector),
      make_not_null(&normal_spacetime_one_form),
      make_not_null(&da_spacetime_metric), d_spacetime_metric, d_pi, d_phi,
      spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
      spacetime_deriv_gauge_function);
}

void compute_spectre_result(
    gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
    const d_spacetime_metric_type& d_spacetime_metric, const d_pi_type& d_pi,
    const d_phi_type& d_phi, const spacetime_metric_type& spacetime_metric,
    const pi_type& pi, const phi_type& phi, const gamma0_type& gamma0,
    const gamma1_type& gamma1, const gamma2_type& gamma2,
    const gauge_function_type& gauge_function,
    const spacetime_deriv_gauge_function_type&
        spacetime_deriv_gauge_function) noexcept {
  // create tensors to be filled by implementation
  // dt_spacetime_metric_type dt_spacetime_metric(num_grid_points);
  dt_pi_type dt_pi(num_grid_points);
  dt_phi_type dt_phi(num_grid_points);
  temp_gamma1_type temp_gamma1(num_grid_points);
  temp_gamma2_type temp_gamma2(num_grid_points);
  gamma1gamma2_type gamma1gamma2(num_grid_points);
  pi_two_normals_type pi_two_normals(num_grid_points);
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint(num_grid_points);
  gamma1_plus_1_type gamma1_plus_1(num_grid_points);
  pi_one_normal_type pi_one_normal(num_grid_points);
  gauge_constraint_type gauge_constraint(num_grid_points);
  phi_two_normals_type phi_two_normals(num_grid_points);
  shift_dot_three_index_constraint_type shift_dot_three_index_constraint(
      num_grid_points);
  phi_one_normal_type phi_one_normal(num_grid_points);
  pi_2_up_type pi_2_up(num_grid_points);
  three_index_constraint_type three_index_constraint(num_grid_points);
  phi_1_up_type phi_1_up(num_grid_points);
  phi_3_up_type phi_3_up(num_grid_points);
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up(num_grid_points);
  lapse_type lapse(num_grid_points);
  shift_type shift(num_grid_points);
  spatial_metric_type spatial_metric(num_grid_points);
  inverse_spatial_metric_type inverse_spatial_metric(num_grid_points);
  det_spatial_metric_type det_spatial_metric(num_grid_points);
  inverse_spacetime_metric_type inverse_spacetime_metric(num_grid_points);
  christoffel_first_kind_type christoffel_first_kind(num_grid_points);
  christoffel_second_kind_type christoffel_second_kind(num_grid_points);
  trace_christoffel_type trace_christoffel(num_grid_points);
  normal_spacetime_vector_type normal_spacetime_vector(num_grid_points);
  normal_spacetime_one_form_type normal_spacetime_one_form(num_grid_points);
  da_spacetime_metric_type da_spacetime_metric(num_grid_points);

  GeneralizedHarmonic::TimeDerivative<Dim>::apply(
      dt_spacetime_metric, make_not_null(&dt_pi), make_not_null(&dt_phi),
      make_not_null(&temp_gamma1), make_not_null(&temp_gamma2),
      make_not_null(&gamma1gamma2), make_not_null(&pi_two_normals),
      make_not_null(&normal_dot_gauge_constraint),
      make_not_null(&gamma1_plus_1), make_not_null(&pi_one_normal),
      make_not_null(&gauge_constraint), make_not_null(&phi_two_normals),
      make_not_null(&shift_dot_three_index_constraint),
      make_not_null(&phi_one_normal), make_not_null(&pi_2_up),
      make_not_null(&three_index_constraint), make_not_null(&phi_1_up),
      make_not_null(&phi_3_up), make_not_null(&christoffel_first_kind_3_up),
      make_not_null(&lapse), make_not_null(&shift),
      make_not_null(&spatial_metric), make_not_null(&inverse_spatial_metric),
      make_not_null(&det_spatial_metric),
      make_not_null(&inverse_spacetime_metric),
      make_not_null(&christoffel_first_kind),
      make_not_null(&christoffel_second_kind),
      make_not_null(&trace_christoffel),
      make_not_null(&normal_spacetime_vector),
      make_not_null(&normal_spacetime_one_form),
      make_not_null(&da_spacetime_metric), d_spacetime_metric, d_pi, d_phi,
      spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
      spacetime_deriv_gauge_function);
}

void compute_te_result(
    const gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_spacetime_metric,
    const gsl::not_null<dt_pi_type*> dt_pi,
    const gsl::not_null<dt_phi_type*> dt_phi,
    const gsl::not_null<temp_gamma1_type*> temp_gamma1,
    const gsl::not_null<temp_gamma2_type*> temp_gamma2,
    const gsl::not_null<gamma1gamma2_type*> gamma1gamma2,
    const gsl::not_null<pi_two_normals_type*> pi_two_normals,
    const gsl::not_null<normal_dot_gauge_constraint_type*>
        normal_dot_gauge_constraint,
    const gsl::not_null<gamma1_plus_1_type*> gamma1_plus_1,
    const gsl::not_null<pi_one_normal_type*> pi_one_normal,
    const gsl::not_null<gauge_constraint_type*> gauge_constraint,
    const gsl::not_null<phi_two_normals_type*> phi_two_normals,
    const gsl::not_null<shift_dot_three_index_constraint_type*>
        shift_dot_three_index_constraint,
    const gsl::not_null<phi_one_normal_type*> phi_one_normal,
    const gsl::not_null<pi_2_up_type*> pi_2_up,
    const gsl::not_null<three_index_constraint_type*> three_index_constraint,
    const gsl::not_null<phi_1_up_type*> phi_1_up,
    const gsl::not_null<phi_3_up_type*> phi_3_up,
    const gsl::not_null<christoffel_first_kind_3_up_type*>
        christoffel_first_kind_3_up,
    const gsl::not_null<lapse_type*> lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> shift,
    const gsl::not_null<spatial_metric_type*> spatial_metric,
    const gsl::not_null<inverse_spatial_metric_type*> inverse_spatial_metric,
    const gsl::not_null<det_spatial_metric_type*> det_spatial_metric,
    const gsl::not_null<inverse_spacetime_metric_type*>
        inverse_spacetime_metric,
    const gsl::not_null<christoffel_first_kind_type*> christoffel_first_kind,
    const gsl::not_null<christoffel_second_kind_type*> christoffel_second_kind,
    const gsl::not_null<trace_christoffel_type*> trace_christoffel,
    const gsl::not_null<normal_spacetime_vector_type*> normal_spacetime_vector,
    const gsl::not_null<normal_spacetime_one_form_type*>
        normal_spacetime_one_form,
    const gsl::not_null<da_spacetime_metric_type*> da_spacetime_metric,
    const d_spacetime_metric_type& d_spacetime_metric, const d_pi_type& d_pi,
    const d_phi_type& d_phi, const spacetime_metric_type& spacetime_metric,
    const pi_type& pi, const phi_type& phi, const gamma0_type& gamma0,
    const gamma1_type& gamma1, const gamma2_type& gamma2,
    const gauge_function_type& gauge_function,
    const spacetime_deriv_gauge_function_type&
        spacetime_deriv_gauge_function) noexcept {
  // Need constraint damping on interfaces in DG schemes
  *temp_gamma1 = gamma1;
  *temp_gamma2 = gamma2;

  // can't do with TE's yet
  gr::spatial_metric(spatial_metric, spacetime_metric);
  // can't do with TE's yet
  determinant_and_inverse(det_spatial_metric, inverse_spatial_metric,
                          *spatial_metric);
  // can't do with TE's yet
  gr::shift(shift, spacetime_metric, *inverse_spatial_metric);
  // can't do with TE's yet
  gr::lapse(lapse, *shift, spacetime_metric);
  // can't do with TE's yet
  gr::inverse_spacetime_metric(inverse_spacetime_metric, *lapse, *shift,
                               *inverse_spatial_metric);
  // can't do with TE's yet
  GeneralizedHarmonic::spacetime_derivative_of_spacetime_metric(
      da_spacetime_metric, *lapse, *shift, pi, phi);
  // Note: 2nd and 3rd indices of result are symmetric
  // auto christoffel_first_kind = evaluate<ti_c, ti_a, ti_b>(
  //    0.5 * (da_spacetime_metric(ti_a, ti_b, ti_c) + da_spacetime_metric(ti_b,
  //    ti_a, ti_c) -
  //           da_spacetime_metric(ti_c, ti_a, ti_b)));
  // gr::christoffel_first_kind(christoffel_first_kind, *da_spacetime_metric);
  TensorExpressions::evaluate<ti_c, ti_a, ti_b>(
      christoffel_first_kind, 0.5 * ((*da_spacetime_metric)(ti_a, ti_b, ti_c) +
                                     (*da_spacetime_metric)(ti_b, ti_a, ti_c) -
                                     (*da_spacetime_metric)(ti_c, ti_a, ti_b)));

  // Note: 2nd and 3rd indices of result are symmetric
  // auto christoffel_second_kind = evaluate<ti_A, ti_b, ti_c>(
  //    christoffel_first_kind(ti_d, ti_b, ti_c) *
  //    inverse_spacetime_metric(ti_A, ti_D));
  // raise_or_lower_first_index(
  //     christoffel_second_kind, *christoffel_first_kind,
  //     *inverse_spacetime_metric);
  TensorExpressions::evaluate<ti_A, ti_b, ti_c>(
      christoffel_second_kind, (*christoffel_first_kind)(ti_d, ti_b, ti_c) *
                                   (*inverse_spacetime_metric)(ti_A, ti_D));
  // auto trace_christoffel = evaluate<ti_a>(
  //    christoffel_first_kind(ti_a, ti_b, ti_c) *
  //    inverse_spacetime_metric(ti_B, ti_C));
  // trace_last_indices(trace_christoffel, *christoffel_first_kind,
  //                    *inverse_spacetime_metric);
  TensorExpressions::evaluate<ti_a>(
      trace_christoffel, (*christoffel_first_kind)(ti_a, ti_b, ti_c) *
                             (*inverse_spacetime_metric)(ti_B, ti_C));
  // can't do with TE's yet
  gr::spacetime_normal_vector(normal_spacetime_vector, *lapse, *shift);
  // can't do with TE's yet
  gr::spacetime_normal_one_form(normal_spacetime_one_form, *lapse);

  // auto gamma1gamma2 = evaluate(gamma1 * gamma 2);
  get(*gamma1gamma2) = get(gamma1) * get(gamma2);
  // not an eq
  const DataVector& gamma12 = get(*gamma1gamma2);

  // Note: 2nd and 3rd indices of result are symmetric
  // auto phi_1_up = evaluate<ti_I, ti_a, ti_b>(
  //     inverse_spatial_metric(ti_I, ti_J) * // II
  //     phi(ti_j, ti_a, ti_b));              // iaa
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        phi_1_up->get(m, mu, nu) =
            inverse_spatial_metric->get(m, 0) * phi.get(0, mu, nu);
        for (size_t n = 1; n < Dim; ++n) {
          phi_1_up->get(m, mu, nu) +=
              inverse_spatial_metric->get(m, n) * phi.get(n, mu, nu);
        }
      }
    }
  }

  // auto phi_3_up = evaluate<ti_i, ti_a, ti_B>( // iaB
  //     inverse_spacetime_metric(ti_B, ti_C) *  // AA
  //     phi(ti_i, ti_a, ti_c));                 // iaa
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        phi_3_up->get(m, nu, alpha) =
            inverse_spacetime_metric->get(alpha, 0) * phi.get(m, nu, 0);
        for (size_t beta = 1; beta < Dim + 1; ++beta) {
          phi_3_up->get(m, nu, alpha) +=
              inverse_spacetime_metric->get(alpha, beta) * phi.get(m, nu, beta);
        }
      }
    }
  }

  // auto pi_2_up = evaluate<ti_a, ti_B>(        // aB
  //     inverse_spacetime_metric(ti_B, ti_C) *  // AA
  //     pi(ti_a, ti_c));                        // aa
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
      pi_2_up->get(nu, alpha) =
          inverse_spacetime_metric->get(alpha, 0) * pi.get(nu, 0);
      for (size_t beta = 1; beta < Dim + 1; ++beta) {
        pi_2_up->get(nu, alpha) +=
            inverse_spacetime_metric->get(alpha, beta) * pi.get(nu, beta);
      }
    }
  }

  // auto christoffel_first_kind_3_up = evaluate<ti_a, ti_b, ti_C>(  //abC
  //     inverse_spacetime_metric(ti_C, ti_D) *                  // AA
  //     christoffel_first_kind(ti_a, ti_b, ti_d));              // abb
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        christoffel_first_kind_3_up->get(mu, nu, alpha) =
            inverse_spacetime_metric->get(alpha, 0) *
            christoffel_first_kind->get(mu, nu, 0);
        for (size_t beta = 1; beta < Dim + 1; ++beta) {
          christoffel_first_kind_3_up->get(mu, nu, alpha) +=
              inverse_spacetime_metric->get(alpha, beta) *
              christoffel_first_kind->get(mu, nu, beta);
        }
      }
    }
  }

  // auto pi_one_normal = evaluate<ti_a>(  // a
  //     normal_spacetime_vector(ti_B) *   // A
  //     pi(ti_b, ti_a));                  // aa
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    pi_one_normal->get(mu) = get<0>(*normal_spacetime_vector) * pi.get(0, mu);
    for (size_t nu = 1; nu < Dim + 1; ++nu) {
      pi_one_normal->get(mu) +=
          normal_spacetime_vector->get(nu) * pi.get(nu, mu);
    }
  }

  // auto pi_two_normals = evaluate(       // scalar
  //     normal_spacetime_vector(ti_A) *   // A
  //     pi_one_normal(ti_a));             // a
  get(*pi_two_normals) =
      get<0>(*normal_spacetime_vector) * get<0>(*pi_one_normal);
  for (size_t mu = 1; mu < Dim + 1; ++mu) {
    get(*pi_two_normals) +=
        normal_spacetime_vector->get(mu) * pi_one_normal->get(mu);
  }

  // auto phi_one_normal = evaluate<ti_i, ti_a>(  // ia
  //     normal_spacetime_vector(ti_B) *          // A
  //     phi(ti_i, ti_b, ti_a));                  // iaa
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      phi_one_normal->get(n, nu) =
          get<0>(*normal_spacetime_vector) * phi.get(n, 0, nu);
      for (size_t mu = 1; mu < Dim + 1; ++mu) {
        phi_one_normal->get(n, nu) +=
            normal_spacetime_vector->get(mu) * phi.get(n, mu, nu);
      }
    }
  }

  // auto phi_two_normals = evaluate<ti_i, ti_a>( // i
  //     normal_spacetime_vector(ti_A) *          // A
  //     phi_one_normal(ti_i, ti_a));             // ia
  for (size_t n = 0; n < Dim; ++n) {
    phi_two_normals->get(n) =
        get<0>(*normal_spacetime_vector) * phi_one_normal->get(n, 0);
    for (size_t mu = 1; mu < Dim + 1; ++mu) {
      phi_two_normals->get(n) +=
          normal_spacetime_vector->get(mu) * phi_one_normal->get(n, mu);
    }
  }

  // Note: 2nd and 3rd indices of result are symmetric
  // auto three_index_constraint = evaluate<ti_i, ti_a, ti_b>( // iaa
  //     d_spacetime_metric(ti_i, ti_a, ti_b) -                // iaa
  //     phi(ti_i, ti_a, ti_b));                               // iaa
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        three_index_constraint->get(n, mu, nu) =
            d_spacetime_metric.get(n, mu, nu) - phi.get(n, mu, nu);
      }
    }
  }

  // auto gauge_constraint = evaluate<ti_a>( // a
  //     gauge_function(ti_a) +              // a
  //     trace_christoffel(ti_a));           // a
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    gauge_constraint->get(nu) =
        gauge_function.get(nu) + trace_christoffel->get(nu);
  }

  // auto gamma1_plus_1 = evaluate(1.0 + gamma1()); // scalar
  get(*gamma1_plus_1) = 1.0 + gamma1.get();
  // not an eq
  const DataVector& gamma1p1 = get(*gamma1_plus_1);

  // auto shift_dot_three_index_constraint = evaluate<ti_a, ti_b>( // aa
  //     shift(ti_I) +                                             // I
  //     three_index_constraint(ti_i, ti_a, ti_b));                // iaa
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      shift_dot_three_index_constraint->get(mu, nu) =
          get<0>(*shift) * three_index_constraint->get(0, mu, nu);
      for (size_t m = 1; m < Dim; ++m) {
        shift_dot_three_index_constraint->get(mu, nu) +=
            shift->get(m) * three_index_constraint->get(m, mu, nu);
      }
    }
  }

  // Equation for dt_spacetime_metric
  // dt_spacetime_metric : aa
  // lapse: scalar
  // pi: aa
  // gamma1p1 (gamma1_plus_1) : scalar
  // shift_dot_three_index_constraint : aa
  // shift : I
  // phi : iaa
  // auto dt_spacetime_metric = evaluate<ti_a, ti_b> (
  //    -1.0 * lapse() * pi(ti_a, ti_b) +
  //     gamma1_plus_1() * shift_dot_three_index_constraint(ti_a, ti_b) +
  //     + shift(ti_I) * phi(ti_i, ti_a, ti_b)));
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      dt_spacetime_metric->get(mu, nu) = -get(*lapse) * pi.get(mu, nu);
      dt_spacetime_metric->get(mu, nu) +=
          gamma1p1 * shift_dot_three_index_constraint->get(mu, nu);
      for (size_t m = 0; m < Dim; ++m) {
        dt_spacetime_metric->get(mu, nu) += shift->get(m) * phi.get(m, mu, nu);
      }
    }
  }
}

void test_gh_timederivative_impl(
    const d_spacetime_metric_type& d_spacetime_metric, const d_pi_type& d_pi,
    const d_phi_type& d_phi, const spacetime_metric_type& spacetime_metric,
    const pi_type& pi, const phi_type& phi, const gamma0_type& gamma0,
    const gamma1_type& gamma1, const gamma2_type& gamma2,
    const gauge_function_type& gauge_function,
    const spacetime_deriv_gauge_function_type&
        spacetime_deriv_gauge_function) noexcept {
  // Create tensors to be filled by SpECTRE implementation
  dt_spacetime_metric_type dt_spacetime_metric_spectre(num_grid_points);
  dt_pi_type dt_pi_spectre(num_grid_points);
  dt_phi_type dt_phi_spectre(num_grid_points);
  temp_gamma1_type temp_gamma1_spectre(num_grid_points);
  temp_gamma2_type temp_gamma2_spectre(num_grid_points);
  gamma1gamma2_type gamma1gamma2_spectre(num_grid_points);
  pi_two_normals_type pi_two_normals_spectre(num_grid_points);
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint_spectre(
      num_grid_points);
  gamma1_plus_1_type gamma1_plus_1_spectre(num_grid_points);
  pi_one_normal_type pi_one_normal_spectre(num_grid_points);
  gauge_constraint_type gauge_constraint_spectre(num_grid_points);
  phi_two_normals_type phi_two_normals_spectre(num_grid_points);
  shift_dot_three_index_constraint_type
      shift_dot_three_index_constraint_spectre(num_grid_points);
  phi_one_normal_type phi_one_normal_spectre(num_grid_points);
  pi_2_up_type pi_2_up_spectre(num_grid_points);
  three_index_constraint_type three_index_constraint_spectre(num_grid_points);
  phi_1_up_type phi_1_up_spectre(num_grid_points);
  phi_3_up_type phi_3_up_spectre(num_grid_points);
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up_spectre(
      num_grid_points);
  lapse_type lapse_spectre(num_grid_points);
  shift_type shift_spectre(num_grid_points);
  spatial_metric_type spatial_metric_spectre(num_grid_points);
  inverse_spatial_metric_type inverse_spatial_metric_spectre(num_grid_points);
  det_spatial_metric_type det_spatial_metric_spectre(num_grid_points);
  inverse_spacetime_metric_type inverse_spacetime_metric_spectre(
      num_grid_points);
  christoffel_first_kind_type christoffel_first_kind_spectre(num_grid_points);
  christoffel_second_kind_type christoffel_second_kind_spectre(num_grid_points);
  trace_christoffel_type trace_christoffel_spectre(num_grid_points);
  normal_spacetime_vector_type normal_spacetime_vector_spectre(num_grid_points);
  normal_spacetime_one_form_type normal_spacetime_one_form_spectre(
      num_grid_points);
  da_spacetime_metric_type da_spacetime_metric_spectre(num_grid_points);

  // Create tensors to be filled by TensorExpression implementation
  dt_spacetime_metric_type dt_spacetime_metric_te(num_grid_points);
  dt_pi_type dt_pi_te(num_grid_points);
  dt_phi_type dt_phi_te(num_grid_points);
  temp_gamma1_type temp_gamma1_te(num_grid_points);
  temp_gamma2_type temp_gamma2_te(num_grid_points);
  gamma1gamma2_type gamma1gamma2_te(num_grid_points);
  pi_two_normals_type pi_two_normals_te(num_grid_points);
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint_te(
      num_grid_points);
  gamma1_plus_1_type gamma1_plus_1_te(num_grid_points);
  pi_one_normal_type pi_one_normal_te(num_grid_points);
  gauge_constraint_type gauge_constraint_te(num_grid_points);
  phi_two_normals_type phi_two_normals_te(num_grid_points);
  shift_dot_three_index_constraint_type shift_dot_three_index_constraint_te(
      num_grid_points);
  phi_one_normal_type phi_one_normal_te(num_grid_points);
  pi_2_up_type pi_2_up_te(num_grid_points);
  three_index_constraint_type three_index_constraint_te(num_grid_points);
  phi_1_up_type phi_1_up_te(num_grid_points);
  phi_3_up_type phi_3_up_te(num_grid_points);
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up_te(
      num_grid_points);
  lapse_type lapse_te(num_grid_points);
  shift_type shift_te(num_grid_points);
  spatial_metric_type spatial_metric_te(num_grid_points);
  inverse_spatial_metric_type inverse_spatial_metric_te(num_grid_points);
  det_spatial_metric_type det_spatial_metric_te(num_grid_points);
  inverse_spacetime_metric_type inverse_spacetime_metric_te(num_grid_points);
  christoffel_first_kind_type christoffel_first_kind_te(num_grid_points);
  christoffel_second_kind_type christoffel_second_kind_te(num_grid_points);
  trace_christoffel_type trace_christoffel_te(num_grid_points);
  normal_spacetime_vector_type normal_spacetime_vector_te(num_grid_points);
  normal_spacetime_one_form_type normal_spacetime_one_form_te(num_grid_points);
  da_spacetime_metric_type da_spacetime_metric_te(num_grid_points);

  // Compute SpECTRE result
  GeneralizedHarmonic::TimeDerivative<Dim>::apply(
      make_not_null(&dt_spacetime_metric_spectre),
      make_not_null(&dt_pi_spectre), make_not_null(&dt_phi_spectre),
      make_not_null(&temp_gamma1_spectre), make_not_null(&temp_gamma2_spectre),
      make_not_null(&gamma1gamma2_spectre),
      make_not_null(&pi_two_normals_spectre),
      make_not_null(&normal_dot_gauge_constraint_spectre),
      make_not_null(&gamma1_plus_1_spectre),
      make_not_null(&pi_one_normal_spectre),
      make_not_null(&gauge_constraint_spectre),
      make_not_null(&phi_two_normals_spectre),
      make_not_null(&shift_dot_three_index_constraint_spectre),
      make_not_null(&phi_one_normal_spectre), make_not_null(&pi_2_up_spectre),
      make_not_null(&three_index_constraint_spectre),
      make_not_null(&phi_1_up_spectre), make_not_null(&phi_3_up_spectre),
      make_not_null(&christoffel_first_kind_3_up_spectre),
      make_not_null(&lapse_spectre), make_not_null(&shift_spectre),
      make_not_null(&spatial_metric_spectre),
      make_not_null(&inverse_spatial_metric_spectre),
      make_not_null(&det_spatial_metric_spectre),
      make_not_null(&inverse_spacetime_metric_spectre),
      make_not_null(&christoffel_first_kind_spectre),
      make_not_null(&christoffel_second_kind_spectre),
      make_not_null(&trace_christoffel_spectre),
      make_not_null(&normal_spacetime_vector_spectre),
      make_not_null(&normal_spacetime_one_form_spectre),
      make_not_null(&da_spacetime_metric_spectre), d_spacetime_metric, d_pi,
      d_phi, spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
      spacetime_deriv_gauge_function);

  // Compute TensorExpression result
  compute_te_result(
      make_not_null(&dt_spacetime_metric_te), make_not_null(&dt_pi_te),
      make_not_null(&dt_phi_te), make_not_null(&temp_gamma1_te),
      make_not_null(&temp_gamma2_te), make_not_null(&gamma1gamma2_te),
      make_not_null(&pi_two_normals_te),
      make_not_null(&normal_dot_gauge_constraint_te),
      make_not_null(&gamma1_plus_1_te), make_not_null(&pi_one_normal_te),
      make_not_null(&gauge_constraint_te), make_not_null(&phi_two_normals_te),
      make_not_null(&shift_dot_three_index_constraint_te),
      make_not_null(&phi_one_normal_te), make_not_null(&pi_2_up_te),
      make_not_null(&three_index_constraint_te), make_not_null(&phi_1_up_te),
      make_not_null(&phi_3_up_te),
      make_not_null(&christoffel_first_kind_3_up_te), make_not_null(&lapse_te),
      make_not_null(&shift_te), make_not_null(&spatial_metric_te),
      make_not_null(&inverse_spatial_metric_te),
      make_not_null(&det_spatial_metric_te),
      make_not_null(&inverse_spacetime_metric_te),
      make_not_null(&christoffel_first_kind_te),
      make_not_null(&christoffel_second_kind_te),
      make_not_null(&trace_christoffel_te),
      make_not_null(&normal_spacetime_vector_te),
      make_not_null(&normal_spacetime_one_form_te),
      make_not_null(&da_spacetime_metric_te), d_spacetime_metric, d_pi, d_phi,
      spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
      spacetime_deriv_gauge_function);

  // CHECK christoffel_first_kind (abb)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(christoffel_first_kind_spectre.get(a, b, c),
                              christoffel_first_kind_te.get(a, b, c));
      }
    }
  }

  // CHECK christoffel_second_kind (Abb)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(christoffel_second_kind_spectre.get(a, b, c),
                              christoffel_second_kind_te.get(a, b, c));
      }
    }
  }

  // CHECK trace_christoffel (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(trace_christoffel_spectre.get(a),
                          trace_christoffel_te.get(a));
  }

  // CHECK dt_spacetime_metric (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_spectre.get(a, b),
                            dt_spacetime_metric_te.get(a, b));
    }
  }
}

template <typename Generator>
void test_gh_timederivative(
    const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  using gh_tags_list = tmpl::list<gr::Tags::SpacetimeMetric<Dim>,
                                  GeneralizedHarmonic::Tags::Pi<Dim>,
                                  GeneralizedHarmonic::Tags::Phi<Dim>>;

  const size_t num_grid_points_1d = 3;
  const Mesh<Dim> mesh(num_grid_points_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const DataVector used_for_size(mesh.number_of_grid_points());

  Variables<gh_tags_list> evolved_vars(mesh.number_of_grid_points());
  fill_with_random_values(make_not_null(&evolved_vars), generator,
                          make_not_null(&distribution));
  // In order to satisfy the physical requirements on the spacetime metric we
  // compute it from the helper functions that generate a physical lapse, shift,
  // and spatial metric.
  gr::spacetime_metric(
      make_not_null(&get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars)),
      TestHelpers::gr::random_lapse(generator, used_for_size),
      TestHelpers::gr::random_shift<Dim>(generator, used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(generator, used_for_size));

  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      if (i == j) {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 1.0);
      } else {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 0.0);
      }
    }
  }

  const auto partial_derivs =
      partial_derivatives<gh_tags_list>(evolved_vars, mesh, inv_jac);

  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars);
  const auto& phi = get<GeneralizedHarmonic::Tags::Phi<Dim>>(evolved_vars);
  const auto& pi = get<GeneralizedHarmonic::Tags::Pi<Dim>>(evolved_vars);
  const auto& d_spacetime_metric =
      get<Tags::deriv<gr::Tags::SpacetimeMetric<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const auto& d_phi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Phi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const auto& d_pi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Pi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  ;

  const auto gamma0 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const auto gamma1 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const auto gamma2 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const auto gauge_function = make_with_random_values<tnsr::a<DataVector, Dim>>(
      generator, make_not_null(&distribution), used_for_size);
  const auto spacetime_deriv_gauge_function =
      make_with_random_values<tnsr::ab<DataVector, Dim>>(
          generator, make_not_null(&distribution), used_for_size);

  test_gh_timederivative_impl(d_spacetime_metric, d_pi, d_phi, spacetime_metric,
                              pi, phi, gamma0, gamma1, gamma2, gauge_function,
                              spacetime_deriv_gauge_function);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.GHTimeDerivative",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);
  test_gh_timederivative(make_not_null(&generator));
}
