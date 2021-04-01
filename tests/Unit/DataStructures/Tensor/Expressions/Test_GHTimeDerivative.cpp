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

// types not in SpECTRE implementation, but needed by TE implementation since
// TEs can't yet iterate over the spatial components of a spacetime index
using pi_one_normal_spatial_type = tnsr::i<DataVector, Dim>;
using phi_one_normal_spatial_type = tnsr::ij<DataVector, Dim>;

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
    const spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function,
    const gsl::not_null<pi_one_normal_spatial_type*> pi_one_normal_spatial,
    const gsl::not_null<phi_one_normal_spatial_type*>
        phi_one_normal_spatial) noexcept {
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
  // gr::christoffel_first_kind(christoffel_first_kind, *da_spacetime_metric);
  TensorExpressions::evaluate<ti_c, ti_a, ti_b>(
      christoffel_first_kind, 0.5 * ((*da_spacetime_metric)(ti_a, ti_b, ti_c) +
                                     (*da_spacetime_metric)(ti_b, ti_a, ti_c) -
                                     (*da_spacetime_metric)(ti_c, ti_a, ti_b)));

  // raise_or_lower_first_index(christoffel_second_kind,
  // *christoffel_first_kind, *inverse_spacetime_metric);
  TensorExpressions::evaluate<ti_A, ti_b, ti_c>(
      christoffel_second_kind, (*christoffel_first_kind)(ti_d, ti_b, ti_c) *
                                   (*inverse_spacetime_metric)(ti_A, ti_D));
  // trace_last_indices(trace_christoffel, *christoffel_first_kind,
  //                    *inverse_spacetime_metric);
  TensorExpressions::evaluate<ti_a>(
      trace_christoffel, (*christoffel_first_kind)(ti_a, ti_b, ti_c) *
                             (*inverse_spacetime_metric)(ti_B, ti_C));
  // Expanded expression for trace_christoffel
  // TensorExpressions::evaluate<ti_c>(
  //     trace_christoffel, 0.5 * ((*da_spacetime_metric)(ti_a, ti_b, ti_c) +
  //                               (*da_spacetime_metric)(ti_b, ti_a, ti_c) -
  //                               (*da_spacetime_metric)(ti_c, ti_a, ti_b)) *
  //                            (*inverse_spacetime_metric)(ti_A, ti_B));
  //
  // can't do with TE's yet
  gr::spacetime_normal_vector(normal_spacetime_vector, *lapse, *shift);
  // can't do with TE's yet
  gr::spacetime_normal_one_form(normal_spacetime_one_form, *lapse);

  // get(*gamma1gamma2) = get(gamma1) * get(gamma2);
  TensorExpressions::evaluate(gamma1gamma2, gamma1() * gamma2());
  // not used in TE equations
  // const DataVector& gamma12 = get(*gamma1gamma2);

  // for (size_t m = 0; m < Dim; ++m) {
  //   for (size_t mu = 0; mu < Dim + 1; ++mu) {
  //     for (size_t nu = mu; nu < Dim + 1; ++nu) {
  //       phi_1_up->get(m, mu, nu) =
  //           inverse_spatial_metric->get(m, 0) * phi.get(0, mu, nu);
  //       for (size_t n = 1; n < Dim; ++n) {
  //         phi_1_up->get(m, mu, nu) +=
  //             inverse_spatial_metric->get(m, n) * phi.get(n, mu, nu);
  //       }
  //     }
  //   }
  // }
  TensorExpressions::evaluate<ti_I, ti_a, ti_b>(
      phi_1_up, (*inverse_spatial_metric)(ti_I, ti_J) * phi(ti_j, ti_a, ti_b));

  // for (size_t m = 0; m < Dim; ++m) {
  //   for (size_t nu = 0; nu < Dim + 1; ++nu) {
  //     for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
  //       phi_3_up->get(m, nu, alpha) =
  //           inverse_spacetime_metric->get(alpha, 0) * phi.get(m, nu, 0);
  //       for (size_t beta = 1; beta < Dim + 1; ++beta) {
  //         phi_3_up->get(m, nu, alpha) +=
  //             inverse_spacetime_metric->get(alpha, beta) *
  //                 phi.get(m, nu, beta);
  //       }
  //     }
  //   }
  // }
  TensorExpressions::evaluate<ti_i, ti_a, ti_B>(
      phi_3_up,
      (*inverse_spacetime_metric)(ti_B, ti_C) * phi(ti_i, ti_a, ti_c));

  // for (size_t nu = 0; nu < Dim + 1; ++nu) {
  //   for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
  //     pi_2_up->get(nu, alpha) =
  //         inverse_spacetime_metric->get(alpha, 0) * pi.get(nu, 0);
  //     for (size_t beta = 1; beta < Dim + 1; ++beta) {
  //       pi_2_up->get(nu, alpha) +=
  //           inverse_spacetime_metric->get(alpha, beta) * pi.get(nu, beta);
  //     }
  //   }
  // }
  TensorExpressions::evaluate<ti_a, ti_B>(
      pi_2_up, (*inverse_spacetime_metric)(ti_B, ti_C) * pi(ti_a, ti_c));

  // for (size_t mu = 0; mu < Dim + 1; ++mu) {
  //   for (size_t nu = 0; nu < Dim + 1; ++nu) {
  //     for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
  //       christoffel_first_kind_3_up->get(mu, nu, alpha) =
  //           inverse_spacetime_metric->get(alpha, 0) *
  //           christoffel_first_kind->get(mu, nu, 0);
  //       for (size_t beta = 1; beta < Dim + 1; ++beta) {
  //         christoffel_first_kind_3_up->get(mu, nu, alpha) +=
  //             inverse_spacetime_metric->get(alpha, beta) *
  //             christoffel_first_kind->get(mu, nu, beta);
  //       }
  //     }
  //   }
  // }
  TensorExpressions::evaluate<ti_a, ti_b, ti_C>(
      christoffel_first_kind_3_up,
      (*inverse_spacetime_metric)(ti_C, ti_D) *
          (*christoffel_first_kind)(ti_a, ti_b, ti_d));

  // for (size_t mu = 0; mu < Dim + 1; ++mu) {
  //   pi_one_normal->get(mu) = get<0>(*normal_spacetime_vector)
  //                                * pi.get(0, mu);
  //   for (size_t nu = 1; nu < Dim + 1; ++nu) {
  //     pi_one_normal->get(mu) +=
  //         normal_spacetime_vector->get(nu) * pi.get(nu, mu);
  //   }
  // }
  TensorExpressions::evaluate<ti_a>(
      pi_one_normal, (*normal_spacetime_vector)(ti_B)*pi(ti_b, ti_a));

  // can't get spatial components of spacetime indices with TE's yet
  // (i.e. the m + 1), so copying over spatial components of pi_one_normal into
  // a new tensor
  for (size_t m = 0; m < Dim; ++m) {
    pi_one_normal_spatial->get(m) = pi_one_normal->get(m + 1);
  }

  // get(*pi_two_normals) =
  //     get<0>(*normal_spacetime_vector) * get<0>(*pi_one_normal);
  // for (size_t mu = 1; mu < Dim + 1; ++mu) {
  //   get(*pi_two_normals) +=
  //       normal_spacetime_vector->get(mu) * pi_one_normal->get(mu);
  // }
  TensorExpressions::evaluate(pi_two_normals, (*normal_spacetime_vector)(ti_A) *
                                                  (*pi_one_normal)(ti_a));

  // for (size_t n = 0; n < Dim; ++n) {
  //   for (size_t nu = 0; nu < Dim + 1; ++nu) {
  //     phi_one_normal->get(n, nu) =
  //         get<0>(*normal_spacetime_vector) * phi.get(n, 0, nu);
  //     for (size_t mu = 1; mu < Dim + 1; ++mu) {
  //       phi_one_normal->get(n, nu) +=
  //           normal_spacetime_vector->get(mu) * phi.get(n, mu, nu);
  //     }
  //   }
  // }
  TensorExpressions::evaluate<ti_i, ti_a>(
      phi_one_normal, (*normal_spacetime_vector)(ti_B)*phi(ti_i, ti_b, ti_a));
  // can't get spatial components of spacetime indices with TE's yet
  // (i.e. the m + 1), so copying over spatial components of phi_one_normal into
  // a new tensor
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      phi_one_normal_spatial->get(i, j) = phi_one_normal->get(i, j + 1);
    }
  }

  // for (size_t n = 0; n < Dim; ++n) {
  //   phi_two_normals->get(n) =
  //       get<0>(*normal_spacetime_vector) * phi_one_normal->get(n, 0);
  //   for (size_t mu = 1; mu < Dim + 1; ++mu) {
  //     phi_two_normals->get(n) +=
  //         normal_spacetime_vector->get(mu) * phi_one_normal->get(n, mu);
  //   }
  // }
  TensorExpressions::evaluate<ti_i>(
      phi_two_normals,
      (*normal_spacetime_vector)(ti_A) * (*phi_one_normal)(ti_i, ti_a));

  // for (size_t n = 0; n < Dim; ++n) {
  //   for (size_t mu = 0; mu < Dim + 1; ++mu) {
  //     for (size_t nu = mu; nu < Dim + 1; ++nu) {
  //       three_index_constraint->get(n, mu, nu) =
  //           d_spacetime_metric.get(n, mu, nu) - phi.get(n, mu, nu);
  //     }
  //   }
  // }
  TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
      three_index_constraint,
      d_spacetime_metric(ti_i, ti_a, ti_b) - phi(ti_i, ti_a, ti_b));

  // for (size_t nu = 0; nu < Dim + 1; ++nu) {
  //   gauge_constraint->get(nu) =
  //       gauge_function.get(nu) + trace_christoffel->get(nu);
  // }
  TensorExpressions::evaluate<ti_a>(
      gauge_constraint, gauge_function(ti_a) + (*trace_christoffel)(ti_a));

  // get(*normal_dot_gauge_constraint) =
  //     get<0>(*normal_spacetime_vector) * get<0>(*gauge_constraint);
  // for (size_t mu = 1; mu < Dim + 1; ++mu) {
  //   get(*normal_dot_gauge_constraint) +=
  //       normal_spacetime_vector->get(mu) * gauge_constraint->get(mu);
  // }
  TensorExpressions::evaluate(
      normal_dot_gauge_constraint,
      (*normal_spacetime_vector)(ti_A) * (*gauge_constraint)(ti_a));

  // get(*gamma1_plus_1) = 1.0 + gamma1.get();
  TensorExpressions::evaluate(gamma1_plus_1, 1.0 + gamma1());
  // not used in TE equations
  // const DataVector& gamma1p1 = get(*gamma1_plus_1);

  // for (size_t mu = 0; mu < Dim + 1; ++mu) {
  //   for (size_t nu = mu; nu < Dim + 1; ++nu) {
  //     shift_dot_three_index_constraint->get(mu, nu) =
  //         get<0>(*shift) * three_index_constraint->get(0, mu, nu);
  //     for (size_t m = 1; m < Dim; ++m) {
  //       shift_dot_three_index_constraint->get(mu, nu) +=
  //           shift->get(m) * three_index_constraint->get(m, mu, nu);
  //     }
  //   }
  // }
  TensorExpressions::evaluate<ti_a, ti_b>(
      shift_dot_three_index_constraint,
      (*shift)(ti_I) * (*three_index_constraint)(ti_i, ti_a, ti_b));

  // Here are the actual equations

  // Equation for dt_spacetime_metric
  // dt_spacetime_metric : aa
  // lapse: scalar
  // pi: aa
  // gamma1p1 (gamma1_plus_1) : scalar
  // shift_dot_three_index_constraint : aa
  // shift : I
  // phi : iaa
  //
  // for (size_t mu = 0; mu < Dim + 1; ++mu) {
  //   for (size_t nu = mu; nu < Dim + 1; ++nu) {
  //     dt_spacetime_metric->get(mu, nu) = -get(*lapse) * pi.get(mu, nu);
  //     dt_spacetime_metric->get(mu, nu) +=
  //         gamma1p1 * shift_dot_three_index_constraint->get(mu, nu);
  //     for (size_t m = 0; m < Dim; ++m) {
  //       dt_spacetime_metric->get(mu, nu) += shift->get(m) * phi.get(m, mu,
  //       nu);
  //     }
  //   }
  // }
  //
  // Written using all terms thus far:
  TensorExpressions::evaluate<ti_a, ti_b> (dt_spacetime_metric,
      -1.0 * (*lapse)() * pi(ti_a, ti_b) +
       (*gamma1_plus_1)() * (*shift_dot_three_index_constraint)(ti_a, ti_b) +
       (*shift)(ti_I) * phi(ti_i, ti_a, ti_b));
  //
  // Written with all expandable terms expanded :
  /*TensorExpressions::evaluate<ti_a, ti_b>(
      dt_spacetime_metric,
      -1.0 * (*lapse)() * pi(ti_a, ti_b) +
          (1.0 + gamma1()) * (*shift)(ti_K) *
              (d_spacetime_metric(ti_k, ti_a, ti_b) - phi(ti_k, ti_a, ti_b)) +
          (*shift)(ti_I) * phi(ti_i, ti_a, ti_b));*/

  // Note: can't do with TE's yet - using pi_one_normal_spatial to enable it
  //
  // Equation for dt_pi
  // dt_pi : aa
  // spacetime_deriv_gauge_function : ab
  // pi_two_normals : scalar
  // pi : aa
  // gamma0 : scalar
  // normal_spacetime_one_form : a
  // gauge_constraint : a
  // spacetime_metric : aa
  // normal_dot_gauge_constraint : scalar
  // christoffel_second_kind :  Abb
  // gauge_function : a
  // pi_2_up : aB
  // phi_1_up : Iaa
  // phi_3_up : iaB
  // christoffel_first_kind_3_up : abC
  // pi_one_normal : a
  // inverse_spatial_metric : II
  // d_phi : ijaa
  // lapse: scalar
  // gamma1gamma2 : scalar
  // shift_dot_three_index_constraint : aa
  // shift : I
  // d_pi : iaa
  //
  // for (size_t mu = 0; mu < Dim + 1; ++mu) {
  //   for (size_t nu = mu; nu < Dim + 1; ++nu) {
  //     dt_pi->get(mu, nu) =
  //         -spacetime_deriv_gauge_function.get(mu, nu) -
  //         spacetime_deriv_gauge_function.get(nu, mu) -
  //         0.5 * get(*pi_two_normals) * pi.get(mu, nu) +
  //         get(gamma0) *
  //             (normal_spacetime_one_form->get(mu) * gauge_constraint->get(nu)
  //             +
  //              normal_spacetime_one_form->get(nu) *
  //              gauge_constraint->get(mu)) -
  //         get(gamma0) * spacetime_metric.get(mu, nu) *
  //             get(*normal_dot_gauge_constraint);

  //     for (size_t delta = 0; delta < Dim + 1; ++delta) {
  //       dt_pi->get(mu, nu) += 2 * christoffel_second_kind->get(delta, mu, nu)
  //       *
  //                                 gauge_function.get(delta) -
  //                             2 * pi.get(mu, delta) * pi_2_up->get(nu,
  //                             delta);

  //       for (size_t n = 0; n < Dim; ++n) {
  //         dt_pi->get(mu, nu) +=
  //             2 * phi_1_up->get(n, mu, delta) * phi_3_up->get(n, nu, delta);
  //       }

  //       for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
  //         dt_pi->get(mu, nu) -=
  //             2. * christoffel_first_kind_3_up->get(mu, alpha, delta) *
  //             christoffel_first_kind_3_up->get(nu, delta, alpha);
  //       }
  //     }

  //     for (size_t m = 0; m < Dim; ++m) {
  //       // can't do with TE's yet
  //       dt_pi->get(mu, nu) -=
  //           pi_one_normal->get(m + 1) * phi_1_up->get(m, mu, nu);

  //       for (size_t n = 0; n < Dim; ++n) {
  //         dt_pi->get(mu, nu) -=
  //             inverse_spatial_metric->get(m, n) * d_phi.get(m, n, mu, nu);
  //       }
  //     }

  //     dt_pi->get(mu, nu) *= get(*lapse);

  //     dt_pi->get(mu, nu) +=
  //         gamma12 * shift_dot_three_index_constraint->get(mu, nu);

  //     for (size_t m = 0; m < Dim; ++m) {
  //       // DualFrame term
  //       dt_pi->get(mu, nu) += shift->get(m) * d_pi.get(m, mu, nu);
  //     }
  //   }
  // }
  //
  // Written using all terms thus far :
  // Note: Whole file compiles in ~30 seconds when this one is used
  TensorExpressions::evaluate<ti_a, ti_b>(
      dt_pi,
      ((-1.0 * spacetime_deriv_gauge_function(ti_a, ti_b)) -
       spacetime_deriv_gauge_function(ti_b, ti_a) -
       0.5 * (*pi_two_normals)() * pi(ti_a, ti_b) +
       gamma0() *
           ((*normal_spacetime_one_form)(ti_a) * (*gauge_constraint)(ti_b) +
            (*normal_spacetime_one_form)(ti_b) * (*gauge_constraint)(ti_a)) -
       gamma0() * spacetime_metric(ti_a, ti_b) *
           (*normal_dot_gauge_constraint)() +
       2.0 * (*christoffel_second_kind)(ti_C, ti_a, ti_b) *
           gauge_function(ti_c) -
       2.0 * pi(ti_a, ti_c) * (*pi_2_up)(ti_b, ti_C) +
       2.0 * (*phi_1_up)(ti_I, ti_a, ti_c) * (*phi_3_up)(ti_i, ti_b, ti_C) -
       2.0 * (*christoffel_first_kind_3_up)(ti_a, ti_d, ti_C) *
           (*christoffel_first_kind_3_up)(ti_b, ti_c, ti_D) -
       (*pi_one_normal_spatial)(ti_j) * (*phi_1_up)(ti_J, ti_a, ti_b) -
       (*inverse_spatial_metric)(ti_J, ti_K) * d_phi(ti_j, ti_k, ti_a, ti_b)) *
              (*lapse)() +
          (*gamma1gamma2)() * (*shift_dot_three_index_constraint)(ti_a, ti_b) +
          (*shift)(ti_I)*d_pi(ti_i, ti_a, ti_b));
  //
  // Written with all expandable terms expanded :
  // Note: Whole file takes ~35-40 min to compile with clang with -j4...
  /*TensorExpressions::evaluate<ti_a, ti_b>(
      dt_pi,
      ((-1.0 * spacetime_deriv_gauge_function(ti_a, ti_b)) -
       spacetime_deriv_gauge_function(ti_b, ti_a) -
       0.5 * (*normal_spacetime_vector)(ti_C) *
           (*normal_spacetime_vector)(ti_D)*pi(ti_c, ti_d) * pi(ti_a, ti_b) +
       gamma0() * ((*normal_spacetime_one_form)(ti_a) *
                       // gauge_function(ti_b) + (*trace_christoffel)(ti_b)
                       (gauge_function(ti_b) +
                        ((0.5 * ((*da_spacetime_metric)(ti_c, ti_d, ti_b) +
                                 (*da_spacetime_metric)(ti_d, ti_c, ti_b) -
                                 (*da_spacetime_metric)(ti_b, ti_c, ti_d))) *
                         (*inverse_spacetime_metric)(ti_C, ti_D))) +
                   (*normal_spacetime_one_form)(ti_b) *
                       // gauge_function(ti_a) + (*trace_christoffel)(ti_a)
                       (gauge_function(ti_a) +
                        ((0.5 * ((*da_spacetime_metric)(ti_c, ti_d, ti_a) +
                                 (*da_spacetime_metric)(ti_d, ti_c, ti_a) -
                                 (*da_spacetime_metric)(ti_a, ti_c, ti_d))) *
                         (*inverse_spacetime_metric)(ti_C, ti_D)))) -
       gamma0() * spacetime_metric(ti_a, ti_b) *
           ((*normal_spacetime_vector)(ti_C) *
            // gauge_function(ti_c) + (*trace_christoffel)(ti_c)
            (gauge_function(ti_c) +
             (((0.5 * ((*da_spacetime_metric)(ti_d, ti_e, ti_c) +
                       (*da_spacetime_metric)(ti_e, ti_d, ti_c) -
                       (*da_spacetime_metric)(ti_c, ti_d, ti_e))) *
               (*inverse_spacetime_metric)(ti_D, ti_E))))) +
       // 2.0 * (*christoffel_first_kind)(ti_d, ti_a, ti_b) *
       //   (*inverse_spacetime_metric)(ti_C, ti_D) *
       2.0 *
           ((((0.5 * ((*da_spacetime_metric)(ti_a, ti_b, ti_d) +
                      (*da_spacetime_metric)(ti_b, ti_a, ti_d) -
                      (*da_spacetime_metric)(ti_d, ti_a, ti_b))) *
              (*inverse_spacetime_metric)(ti_C, ti_D)))) *
           gauge_function(ti_c) -
       2.0 * pi(ti_a, ti_c) * (*inverse_spacetime_metric)(ti_C, ti_D) *
           pi(ti_b, ti_d) +
       2.0 * (*inverse_spatial_metric)(ti_I, ti_J) * phi(ti_j, ti_a, ti_c) *
           (*inverse_spacetime_metric)(ti_C, ti_D) * phi(ti_i, ti_b, ti_d) -
       // 2.0 * (*inverse_spacetime_metric)(ti_C, ti_E) *
       //   (*christoffel_first_kind)(ti_a, ti_d, ti_e) *
       2.0 * (*inverse_spacetime_metric)(ti_C, ti_E) *
           (0.5 * ((*da_spacetime_metric)(ti_d, ti_e, ti_a) +
                   (*da_spacetime_metric)(ti_e, ti_d, ti_a) -
                   (*da_spacetime_metric)(ti_a, ti_d, ti_e))) *
           // (*inverse_spacetime_metric)(ti_D, ti_F) *
           //   (*christoffel_first_kind)(ti_b, ti_c, ti_f) -
           (*inverse_spacetime_metric)(ti_D, ti_F) *
           (0.5 * ((*da_spacetime_metric)(ti_c, ti_f, ti_b) +
                   (*da_spacetime_metric)(ti_f, ti_c, ti_b) -
                   (*da_spacetime_metric)(ti_b, ti_c, ti_f))) -
       (*pi_one_normal_spatial)(ti_j) * (*inverse_spatial_metric)(ti_J, ti_I) *
           phi(ti_i, ti_a, ti_b) -
       (*inverse_spatial_metric)(ti_J, ti_K) * d_phi(ti_j, ti_k, ti_a, ti_b)) *
              (*lapse)() +
          gamma1() * gamma2() * (*shift)(ti_I) *
              (d_spacetime_metric(ti_i, ti_a, ti_b) - phi(ti_i, ti_a, ti_b)) +
          (*shift)(ti_I)*d_pi(ti_i, ti_a, ti_b));*/
  //
  // Written with terms seen in equation reference by SpECTRE documentation :
  // (i.e. some terms in the above fully expanded version have been collapsed)
  // Note: Whole file takes ~18 min to compile with clang with -j4...
  /*TensorExpressions::evaluate<ti_a, ti_b>(
      dt_pi,
      ((-1.0 * spacetime_deriv_gauge_function(ti_a, ti_b)) -
       spacetime_deriv_gauge_function(ti_b, ti_a) -
       0.5 * (*normal_spacetime_vector)(ti_C) *
           (*normal_spacetime_vector)(ti_D)*pi(ti_c, ti_d) * pi(ti_a, ti_b) +
       gamma0() * ((*normal_spacetime_one_form)(ti_a) *
                       (gauge_function(ti_b) + (*trace_christoffel)(ti_b)) +
                   (*normal_spacetime_one_form)(ti_b) *
                       (gauge_function(ti_a) + (*trace_christoffel)(ti_a))) -
       gamma0() * spacetime_metric(ti_a, ti_b) *
           (*normal_spacetime_vector)(ti_C) *
            (gauge_function(ti_c) + (*trace_christoffel)(ti_c)) +
       2.0 * (*christoffel_first_kind)(ti_d, ti_a, ti_b) *
         (*inverse_spacetime_metric)(ti_C, ti_D) *
           gauge_function(ti_c) -
       2.0 * pi(ti_a, ti_c) * (*inverse_spacetime_metric)(ti_C, ti_D) *
           pi(ti_b, ti_d) +
       2.0 * (*inverse_spatial_metric)(ti_I, ti_J) * phi(ti_j, ti_a, ti_c) *
           (*inverse_spacetime_metric)(ti_C, ti_D) * phi(ti_i, ti_b, ti_d) -
       2.0 * (*inverse_spacetime_metric)(ti_C, ti_E) *
         (*christoffel_first_kind)(ti_a, ti_d, ti_e) *
           (*inverse_spacetime_metric)(ti_D, ti_F) *
             (*christoffel_first_kind)(ti_b, ti_c, ti_f) -
       (*pi_one_normal_spatial)(ti_j) * (*inverse_spatial_metric)(ti_J, ti_I) *
           phi(ti_i, ti_a, ti_b) -
       (*inverse_spatial_metric)(ti_J, ti_K) * d_phi(ti_j, ti_k, ti_a, ti_b)) *
              (*lapse)() +
          gamma1() * gamma2() * (*shift)(ti_I) *
              (d_spacetime_metric(ti_i, ti_a, ti_b) - phi(ti_i, ti_a, ti_b)) +
          (*shift)(ti_I)*d_pi(ti_i, ti_a, ti_b));*/

  // Note: can't do with TE's yet - using phi_one_normal_spatial to enable it
  //
  // Equation for dt_phi
  // dt_phi : iaa
  // phi_two_normals : i
  // d_pi : iaa
  // gamma2 : scalar
  // three_index_constraint : iaa
  // phi_one_normal : ia
  // phi_1_up : Iaa
  // lapse: scalar
  // shift : I
  // d_phi : ijaa
  // for (size_t i = 0; i < Dim; ++i) {
  //   for (size_t mu = 0; mu < Dim + 1; ++mu) {
  //     for (size_t nu = mu; nu < Dim + 1; ++nu) {
  //       dt_phi->get(i, mu, nu) =
  //           0.5 * pi.get(mu, nu) * phi_two_normals->get(i) -
  //           d_pi.get(i, mu, nu) +
  //           get(gamma2) * three_index_constraint->get(i, mu, nu);
  //       for (size_t n = 0; n < Dim; ++n) {
  //         // can't do with TE's yet
  //         dt_phi->get(i, mu, nu) +=
  //             phi_one_normal->get(i, n + 1) * phi_1_up->get(n, mu, nu);
  //       }

  //       dt_phi->get(i, mu, nu) *= get(*lapse);
  //       for (size_t m = 0; m < Dim; ++m) {
  //         dt_phi->get(i, mu, nu) += shift->get(m) * d_phi.get(m, i, mu, nu);
  //       }
  //     }
  //   }
  // }
  //
  // Written using all terms thus far :
  TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
      dt_phi,
      (0.5 * pi(ti_a, ti_b) * (*phi_two_normals)(ti_i)-d_pi(ti_i, ti_a, ti_b) +
       gamma2() * (*three_index_constraint)(ti_i, ti_a, ti_b) +
       (*phi_one_normal_spatial)(ti_i, ti_j) * (*phi_1_up)(ti_J, ti_a, ti_b)) *
              (*lapse)() +
          (*shift)(ti_K)*d_phi(ti_k, ti_i, ti_a, ti_b));
  //
  // Written with all expandable terms expanded :
  /*TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
      dt_phi,
      (0.5 * pi(ti_a, ti_b) * (*normal_spacetime_vector)(ti_C) *
           (*normal_spacetime_vector)(ti_D)*phi(ti_i, ti_d, ti_c) -
       d_pi(ti_i, ti_a, ti_b) +
       gamma2() *
           (d_spacetime_metric(ti_i, ti_a, ti_b) - phi(ti_i, ti_a, ti_b)) +
       (*phi_one_normal_spatial)(ti_i, ti_j) *
           (*inverse_spatial_metric)(ti_J, ti_K) * phi(ti_k, ti_a, ti_b)) *
              (*lapse)() +
          (*shift)(ti_K)*d_phi(ti_k, ti_i, ti_a, ti_b));*/
  //
  // Written with all expandable terms rearranged to match solving for dt_phi
  // equation referenced by SpECTRE documentation :
  /*TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
      dt_phi,
      (*lapse)() * (0.5 * (*normal_spacetime_vector)(ti_C) *
                        (*normal_spacetime_vector)(ti_D)*phi(ti_i, ti_d, ti_c) *
                        pi(ti_a, ti_b) +
                    (*inverse_spatial_metric)(ti_J, ti_K) *
                        (*phi_one_normal_spatial)(ti_i, ti_j) *
                        phi(ti_k, ti_a, ti_b) +
                    gamma2() * (d_spacetime_metric(ti_i, ti_a, ti_b) -
                                phi(ti_i, ti_a, ti_b)) -
                    d_pi(ti_i, ti_a, ti_b)) +
          (*shift)(ti_K)*d_phi(ti_k, ti_i, ti_a, ti_b));*/
}

void test_gh_timederivative_impl(
    const size_t num_grid_points,
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
  // TEs can't iterate over only spatial indices of a spacetime index yet, so it
  // will be manually computed to enable writing the equation for dt_pi
  pi_one_normal_spatial_type pi_one_normal_spatial(num_grid_points);
  phi_one_normal_spatial_type phi_one_normal_spatial(num_grid_points);

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
      spacetime_deriv_gauge_function, make_not_null(&pi_one_normal_spatial),
      make_not_null(&phi_one_normal_spatial));

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

  // CHECK gamma1gamma2 (scalar)
  CHECK_ITERABLE_APPROX(gamma1gamma2_spectre.get(), gamma1gamma2_te.get());

  // CHECK phi_1_up (Iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_1_up_spectre.get(i, a, b),
                              phi_1_up_te.get(i, a, b));
      }
    }
  }

  // CHECK phi_3_up (iaB)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_3_up_spectre.get(i, a, b),
                              phi_3_up_te.get(i, a, b));
      }
    }
  }

  // CHECK pi_2_up (aB)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(pi_2_up_spectre.get(a, b), pi_2_up_te.get(a, b));
    }
  }

  // CHECK christoffel_first_kind_3_up (abC)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(christoffel_first_kind_3_up_spectre.get(a, b, c),
                              christoffel_first_kind_3_up_te.get(a, b, c));
      }
    }
  }

  // CHECK pi_one_normal (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(pi_one_normal_spectre.get(a),
                          pi_one_normal_te.get(a));
  }

  // CHECK pi_two_normals (scalar)
  CHECK_ITERABLE_APPROX(pi_two_normals_spectre.get(), pi_two_normals_te.get());

  // CHECK phi_one_normal (ia)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      CHECK_ITERABLE_APPROX(phi_one_normal_spectre.get(i, a),
                            phi_one_normal_te.get(i, a));
    }
  }

  // CHECK phi_two_normals (i)
  for (size_t i = 0; i < Dim; i++) {
    CHECK_ITERABLE_APPROX(phi_two_normals_spectre.get(i),
                          phi_two_normals_te.get(i));
  }

  // CHECK three_index_constraint (iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(three_index_constraint_spectre.get(i, a, b),
                              three_index_constraint_te.get(i, a, b));
      }
    }
  }

  // CHECK gauge_constraint (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(gauge_constraint_spectre.get(a),
                          gauge_constraint_te.get(a));
  }

  // CHECK normal_dot_gauge_constraint (scalar)
  CHECK_ITERABLE_APPROX(normal_dot_gauge_constraint_spectre.get(),
                        normal_dot_gauge_constraint_te.get());

  // CHECK gamma1_plus_1 (scalar)
  CHECK_ITERABLE_APPROX(gamma1_plus_1_spectre.get(), gamma1_plus_1_te.get());

  // CHECK shift_dot_three_index_constraint (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(shift_dot_three_index_constraint_spectre.get(a, b),
                            shift_dot_three_index_constraint_te.get(a, b));
    }
  }

  // CHECK dt_spacetime_metric (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_spectre.get(a, b),
                            dt_spacetime_metric_te.get(a, b));
    }
  }

  // CHECK dt_pi (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_pi_spectre.get(a, b), dt_pi_te.get(a, b));
    }
  }

  // CHECK dt_phi (iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(dt_phi_spectre.get(i, a, b),
                              dt_phi_te.get(i, a, b));
      }
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

  test_gh_timederivative_impl(mesh.number_of_grid_points(), d_spacetime_metric,
                              d_pi, d_phi, spacetime_metric, pi, phi, gamma0,
                              gamma1, gamma2, gauge_function,
                              spacetime_deriv_gauge_function);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.GHTimeDerivative",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);
  test_gh_timederivative(make_not_null(&generator));
}
