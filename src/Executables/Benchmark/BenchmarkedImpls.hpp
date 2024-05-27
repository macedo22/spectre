// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
// #include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

// Implementations benchmarked
template <typename Tensor>
void fill_with_values(const gsl::not_null<Tensor*> tensor) {
 double value = 1.0;
 for (size_t i = 0; i < tensor->size(); i++) {
    for (size_t elem_index = 0; elem_index < (*tensor)[0].size(); elem_index++) {
      (*tensor)[i][elem_index] = value;
      value = -value*1.01;
    }
 }
}

template<typename DataType, size_t Dim>
struct BenchmarkImpl {
  // tensor types in tensor equation being benchmarked
  using dt_spacetime_metric_type = tnsr::aa<DataType, Dim>;
  using dt_pi_type = tnsr::aa<DataType, Dim>;
  using dt_phi_type = tnsr::iaa<DataType, Dim>;
  using temp_gamma1_type = Scalar<DataType>;
  using temp_gamma2_type = Scalar<DataType>;
  using temp_gauge_function_type = tnsr::a<DataType, Dim>;
  using temp_spacetime_deriv_gauge_function_type = tnsr::ab<DataVector, Dim>;
  using gamma1gamma2_type = Scalar<DataType>;
  using half_pi_two_normals_type = Scalar<DataType>;
  using normal_dot_gauge_constraint_type = Scalar<DataType>;
  using gamma1_plus_1_type = Scalar<DataType>;
  using pi_one_normal_type = tnsr::a<DataType, Dim>;
  using gauge_constraint_type = tnsr::a<DataType, Dim>;
  using half_phi_two_normals_type = tnsr::i<DataType, Dim>;
  using shift_dot_three_index_constraint_type = tnsr::aa<DataType, Dim>;
  using mesh_velocity_dot_three_index_constraint_type = tnsr::aa<DataType, Dim>;
  using phi_one_normal_type = tnsr::ia<DataType, Dim>;
  using pi_2_up_type = tnsr::aB<DataType, Dim>;
  using three_index_constraint_type = tnsr::iaa<DataType, Dim>;
  using phi_1_up_type = tnsr::Iaa<DataType, Dim>;
  using phi_3_up_type = tnsr::iaB<DataType, Dim>;
  using christoffel_first_kind_3_up_type = tnsr::abC<DataType, Dim>;
  using lapse_type = Scalar<DataType>;
  using shift_type = tnsr::I<DataType, Dim>;
  using inverse_spatial_metric_type = tnsr::II<DataType, Dim>;
  using det_spatial_metric_type = Scalar<DataType>;
  using sqrt_det_spatial_metric_type = Scalar<DataType>;
  using inverse_spacetime_metric_type = tnsr::AA<DataType, Dim>;
  using christoffel_first_kind_type = tnsr::abb<DataType, Dim>;
  using christoffel_second_kind_type = tnsr::Abb<DataType, Dim>;
  using trace_christoffel_type = tnsr::a<DataVector, Dim>;
  using normal_spacetime_vector_type = tnsr::A<DataType, Dim>;
  using d_spacetime_metric_type = tnsr::iaa<DataType, Dim>;
  using d_pi_type = tnsr::iaa<DataType, Dim>;
  using d_phi_type = tnsr::ijaa<DataType, Dim>;
  using spacetime_metric_type = tnsr::aa<DataType, Dim>;
  using pi_type = tnsr::aa<DataType, Dim>;
  using phi_type = tnsr::iaa<DataType, Dim>;
  using gamma0_type = Scalar<DataType>;
  using gamma1_type = Scalar<DataType>;
  using gamma2_type = Scalar<DataType>;
  using gauge_condition_type = gh::gauges::DampedHarmonic;
  using mesh_type = Mesh<Dim>;
  using inertial_coords_type = tnsr::I<DataVector, Dim, Frame::Inertial>;
  using inverse_jacobian_type = InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>;
  using mesh_velocity_type = tnsr::I<DataType, Dim>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      const gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
      const gsl::not_null<dt_pi_type*> dt_pi,
      const gsl::not_null<dt_phi_type*> dt_phi,
      const gsl::not_null<temp_gamma1_type*> temp_gamma1,
      const gsl::not_null<temp_gamma2_type*> temp_gamma2,
      const gsl::not_null<temp_gauge_function_type*> temp_gauge_function,
      const gsl::not_null<temp_spacetime_deriv_gauge_function_type*>
          temp_spacetime_deriv_gauge_function,
      const gsl::not_null<gamma1gamma2_type*> gamma1gamma2,
      const gsl::not_null<half_pi_two_normals_type*> half_pi_two_normals,
      const gsl::not_null<normal_dot_gauge_constraint_type*>
          normal_dot_gauge_constraint,
      const gsl::not_null<gamma1_plus_1_type*> gamma1_plus_1,
      const gsl::not_null<pi_one_normal_type*> pi_one_normal,
      const gsl::not_null<gauge_constraint_type*> gauge_constraint,
      const gsl::not_null<half_phi_two_normals_type*> half_phi_two_normals,
      const gsl::not_null<shift_dot_three_index_constraint_type*>
          shift_dot_three_index_constraint,
      const gsl::not_null<mesh_velocity_dot_three_index_constraint_type*>
          mesh_velocity_dot_three_index_constraint,
      const gsl::not_null<phi_one_normal_type*> phi_one_normal,
      const gsl::not_null<pi_2_up_type*> pi_2_up,
      const gsl::not_null<three_index_constraint_type*> three_index_constraint,
      const gsl::not_null<phi_1_up_type*> phi_1_up,
      const gsl::not_null<phi_3_up_type*> phi_3_up,
      const gsl::not_null<christoffel_first_kind_3_up_type*>
          christoffel_first_kind_3_up,
      const gsl::not_null<lapse_type*> lapse,
      const gsl::not_null<shift_type*> shift,
      const gsl::not_null<inverse_spatial_metric_type*> inverse_spatial_metric,
      const gsl::not_null<det_spatial_metric_type*> det_spatial_metric,
      const gsl::not_null<sqrt_det_spatial_metric_type*> sqrt_det_spatial_metric,
      const gsl::not_null<inverse_spacetime_metric_type*>
          inverse_spacetime_metric,
      const gsl::not_null<christoffel_first_kind_type*> christoffel_first_kind,
      const gsl::not_null<christoffel_second_kind_type*>
          christoffel_second_kind,
      const gsl::not_null<trace_christoffel_type*> trace_christoffel,
      const gsl::not_null<normal_spacetime_vector_type*>
          normal_spacetime_vector,
      const d_spacetime_metric_type& d_spacetime_metric, const d_pi_type& d_pi,
      const d_phi_type& d_phi, const spacetime_metric_type& spacetime_metric,
      const pi_type& pi, const phi_type& phi, const gamma0_type& gamma0,
      const gamma1_type& gamma1, const gamma2_type& gamma2,
      const gauge_condition_type& gauge_condition,
      const mesh_type& mesh,
      const double time,
      const inertial_coords_type& inertial_coords,
      const inverse_jacobian_type& inverse_jacobian,
      const std::optional<mesh_velocity_type>&
        mesh_velocity) {
    gh::TimeDerivative<Dim>::apply(
        dt_spacetime_metric, dt_pi,
        dt_phi, temp_gamma1,
        temp_gamma2,
        temp_gauge_function,
        temp_spacetime_deriv_gauge_function,
        gamma1gamma2,
        half_pi_two_normals,
        normal_dot_gauge_constraint,
        gamma1_plus_1, pi_one_normal,
        gauge_constraint, half_phi_two_normals,
        shift_dot_three_index_constraint,
        mesh_velocity_dot_three_index_constraint,
        phi_one_normal, pi_2_up,
        three_index_constraint, phi_1_up,
        phi_3_up, christoffel_first_kind_3_up,
        lapse, shift,
        inverse_spatial_metric,
        det_spatial_metric,
        sqrt_det_spatial_metric,
        inverse_spacetime_metric,
        christoffel_first_kind,
        christoffel_second_kind,
        trace_christoffel,
        normal_spacetime_vector,
        d_spacetime_metric, d_pi, d_phi,
        spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_condition,
        mesh, time, inertial_coords, inverse_jacobian, mesh_velocity);
  }

//   SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
//       const gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
//       const gsl::not_null<dt_pi_type*> dt_pi,
//       const gsl::not_null<dt_phi_type*> dt_phi,
//       const gsl::not_null<temp_gamma1_type*> temp_gamma1,
//       const gsl::not_null<temp_gamma2_type*> temp_gamma2,
//       const gsl::not_null<gamma1gamma2_type*> gamma1gamma2,
//       const gsl::not_null<pi_two_normals_type*> pi_two_normals,
//       const gsl::not_null<normal_dot_gauge_constraint_type*>
//           normal_dot_gauge_constraint,
//       const gsl::not_null<gamma1_plus_1_type*> gamma1_plus_1,
//       const gsl::not_null<pi_one_normal_type*> pi_one_normal,
//       const gsl::not_null<gauge_constraint_type*> gauge_constraint,
//       const gsl::not_null<phi_two_normals_type*> phi_two_normals,
//       const gsl::not_null<shift_dot_three_index_constraint_type*>
//           shift_dot_three_index_constraint,
//       const gsl::not_null<phi_one_normal_type*> phi_one_normal,
//       const gsl::not_null<pi_2_up_type*> pi_2_up,
//       const gsl::not_null<three_index_constraint_type*> three_index_constraint,
//       const gsl::not_null<phi_1_up_type*> phi_1_up,
//       const gsl::not_null<phi_3_up_type*> phi_3_up,
//       const gsl::not_null<christoffel_first_kind_3_up_type*>
//           christoffel_first_kind_3_up,
//       const gsl::not_null<lapse_type*> lapse,
//       const gsl::not_null<shift_type*> shift,
//       const gsl::not_null<spatial_metric_type*> spatial_metric,
//       const gsl::not_null<inverse_spatial_metric_type*> inverse_spatial_metric,
//       const gsl::not_null<det_spatial_metric_type*> det_spatial_metric,
//       const gsl::not_null<inverse_spacetime_metric_type*>
//           inverse_spacetime_metric,
//       const gsl::not_null<christoffel_first_kind_type*> christoffel_first_kind,
//       const gsl::not_null<christoffel_second_kind_type*>
//           christoffel_second_kind,
//       const gsl::not_null<trace_christoffel_type*> trace_christoffel,
//       const gsl::not_null<normal_spacetime_vector_type*>
//           normal_spacetime_vector,
//       const gsl::not_null<normal_spacetime_one_form_type*>
//           normal_spacetime_one_form,
//       const gsl::not_null<da_spacetime_metric_type*> da_spacetime_metric,
//       const d_spacetime_metric_type& d_spacetime_metric, const d_pi_type& d_pi,
//       const d_phi_type& d_phi, const spacetime_metric_type& spacetime_metric,
//       const pi_type& pi, const phi_type& phi, const gamma0_type& gamma0,
//       const gamma1_type& gamma1, const gamma2_type& gamma2,
//       const gauge_function_type& gauge_function,
//       const spacetime_deriv_gauge_function_type&
//           spacetime_deriv_gauge_function,
//       const std::optional<mesh_velocity_type>&
//         mesh_velocity) {
//     // Need constraint damping on interfaces in DG schemes
//     *temp_gamma1 = gamma1;
//     *temp_gamma2 = gamma2;

//     gr::spatial_metric(spatial_metric, spacetime_metric);
//     determinant_and_inverse(det_spatial_metric, inverse_spatial_metric,
//                             *spatial_metric);
//     gr::shift(shift, spacetime_metric, *inverse_spatial_metric);
//     gr::lapse(lapse, *shift, spacetime_metric);
//     gr::inverse_spacetime_metric(inverse_spacetime_metric, *lapse, *shift,
//                                  *inverse_spatial_metric);
//     GeneralizedHarmonic::spacetime_derivative_of_spacetime_metric(
//         da_spacetime_metric, *lapse, *shift, pi, phi);
//     tenex::evaluate<ti::c, ti::a, ti::b>(
//         christoffel_first_kind,
//         0.5 * ((*da_spacetime_metric)(ti::a, ti::b, ti::c) +
//                (*da_spacetime_metric)(ti::b, ti::a, ti::c) -
//                (*da_spacetime_metric)(ti::c, ti::a, ti::b)));
//     tenex::evaluate<ti::A, ti::b, ti::c>(
//         christoffel_second_kind, (*christoffel_first_kind)(ti::d, ti::b, ti::c) *
//                                      (*inverse_spacetime_metric)(ti::A, ti::D));
//     tenex::evaluate<ti::a>(
//         trace_christoffel, (*christoffel_first_kind)(ti::a, ti::b, ti::c) *
//                                (*inverse_spacetime_metric)(ti::B, ti::C));
//     gr::spacetime_normal_vector(normal_spacetime_vector, *lapse, *shift);
//     gr::spacetime_normal_one_form(normal_spacetime_one_form, *lapse);

//     tenex::evaluate(gamma1gamma2, gamma1() * gamma2());

//     tenex::evaluate<ti::I, ti::a, ti::b>(
//         phi_1_up,
//         (*inverse_spatial_metric)(ti::I, ti::J) * phi(ti::j, ti::a, ti::b));

//     tenex::evaluate<ti::i, ti::a, ti::B>(
//         phi_3_up,
//         (*inverse_spacetime_metric)(ti::B, ti::C) * phi(ti::i, ti::a, ti::c));

//     tenex::evaluate<ti::a, ti::B>(
//         pi_2_up, (*inverse_spacetime_metric)(ti::B, ti::C) * pi(ti::a, ti::c));

//     tenex::evaluate<ti::a, ti::b, ti::C>(
//         christoffel_first_kind_3_up,
//         (*inverse_spacetime_metric)(ti::C, ti::D) *
//             (*christoffel_first_kind)(ti::a, ti::b, ti::d));

//     tenex::evaluate<ti::a>(
//         pi_one_normal, (*normal_spacetime_vector)(ti::B)*pi(ti::b, ti::a));

//     tenex::evaluate(
//         pi_two_normals,
//         (*normal_spacetime_vector)(ti::A) * (*pi_one_normal)(ti::a));

//     tenex::evaluate<ti::i, ti::a>(
//         phi_one_normal, (*normal_spacetime_vector)(ti::B)*phi(ti::i, ti::b, ti::a));

//     tenex::evaluate<ti::i>(
//         phi_two_normals,
//         (*normal_spacetime_vector)(ti::A) * (*phi_one_normal)(ti::i, ti::a));

//     tenex::evaluate<ti::i, ti::a, ti::b>(
//         three_index_constraint,
//         d_spacetime_metric(ti::i, ti::a, ti::b) - phi(ti::i, ti::a, ti::b));

//     tenex::evaluate<ti::a>(
//         gauge_constraint, gauge_function(ti::a) + (*trace_christoffel)(ti::a));

//     tenex::evaluate(
//         normal_dot_gauge_constraint,
//         (*normal_spacetime_vector)(ti::A) * (*gauge_constraint)(ti::a));

//     tenex::evaluate(gamma1_plus_1, 1.0 + gamma1());

//     tenex::evaluate<ti::a, ti::b>(
//         shift_dot_three_index_constraint,
//         (*shift)(ti::I) * (*three_index_constraint)(ti::i, ti::a, ti::b));

//     // Here are the actual equations

//     // Equation for dt_spacetime_metric
//     tenex::evaluate<ti::a, ti::b>(
//         dt_spacetime_metric,
//         -1.0 * (*lapse)() * pi(ti::a, ti::b) +
//             (*gamma1_plus_1)() *
//                 (*shift_dot_three_index_constraint)(ti::a, ti::b) +
//             (*shift)(ti::I)*phi(ti::i, ti::a, ti::b));

//     // Equation for dt_pi
//     tenex::evaluate<ti::a, ti::b>(
//         dt_pi,
//         ((-1.0 * spacetime_deriv_gauge_function(ti::a, ti::b)) -
//          spacetime_deriv_gauge_function(ti::b, ti::a) -
//          0.5 * (*pi_two_normals)() * pi(ti::a, ti::b) +
//          gamma0() *
//              ((*normal_spacetime_one_form)(ti::a) * (*gauge_constraint)(ti::b) +
//               (*normal_spacetime_one_form)(ti::b) * (*gauge_constraint)(ti::a)) -
//          gamma0() * spacetime_metric(ti::a, ti::b) *
//              (*normal_dot_gauge_constraint)() +
//          2.0 * (*christoffel_second_kind)(ti::C, ti::a, ti::b) *
//              gauge_function(ti::c) -
//          2.0 * pi(ti::a, ti::c) * (*pi_2_up)(ti::b, ti::C) +
//          2.0 * (*phi_1_up)(ti::I, ti::a, ti::c) * (*phi_3_up)(ti::i, ti::b, ti::C) -
//          2.0 * (*christoffel_first_kind_3_up)(ti::a, ti::d, ti::C) *
//              (*christoffel_first_kind_3_up)(ti::b, ti::c, ti::D) -
//          (*pi_one_normal)(ti::j) * (*phi_1_up)(ti::J, ti::a, ti::b) -
//          (*inverse_spatial_metric)(ti::J, ti::K) *
//              d_phi(ti::j, ti::k, ti::a, ti::b)) *
//                 (*lapse)() +
//             (*gamma1gamma2)() *
//                 (*shift_dot_three_index_constraint)(ti::a, ti::b) +
//             (*shift)(ti::I)*d_pi(ti::i, ti::a, ti::b));

//     // Equation for dt_phi
//     tenex::evaluate<ti::i, ti::a, ti::b>(
//         dt_phi,
//         (0.5 * pi(ti::a, ti::b) *
//              (*phi_two_normals)(ti::i)-d_pi(ti::i, ti::a, ti::b) +
//          gamma2() * (*three_index_constraint)(ti::i, ti::a, ti::b) +
//          (*phi_one_normal)(ti::i, ti::j) * (*phi_1_up)(ti::J, ti::a, ti::b)) *
//                 (*lapse)() +
//             (*shift)(ti::K)*d_phi(ti::k, ti::i, ti::a, ti::b));
//   }
};
