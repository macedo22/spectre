// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>
#include <iterator>
#include <random>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Executables/Benchmark/GHTimeDerivative/all/BenchmarkedImpls.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename... Ts>
void copy_tensor(const Tensor<Ts...>& tensor_source,
                 gsl::not_null<Tensor<Ts...>*> tensor_destination) noexcept {
  auto tensor_source_it = tensor_source.begin();
  auto tensor_destination_it = tensor_destination->begin();
  for (; tensor_source_it != tensor_source.end();
       tensor_source_it++, tensor_destination_it++) {
    *tensor_destination_it = *tensor_source_it;
  }
}
}  // namespace

// Make sure TE impl matches manual impl
template <size_t Dim, typename DataType, typename Generator>
void test_benchmarked_impls_core(
    const DataType& used_for_size,
    const gsl::not_null<Generator*> generator) noexcept {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using dt_pi_type = typename BenchmarkImpl::dt_pi_type;
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using temp_gamma1_type = typename BenchmarkImpl::temp_gamma1_type;
  using temp_gamma2_type = typename BenchmarkImpl::temp_gamma2_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using pi_two_normals_type = typename BenchmarkImpl::pi_two_normals_type;
  using normal_dot_gauge_constraint_type =
      typename BenchmarkImpl::normal_dot_gauge_constraint_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using pi_one_normal_type = typename BenchmarkImpl::pi_one_normal_type;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using pi_2_up_type = typename BenchmarkImpl::pi_2_up_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using phi_3_up_type = typename BenchmarkImpl::phi_3_up_type;
  using christoffel_first_kind_3_up_type =
      typename BenchmarkImpl::christoffel_first_kind_3_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using spatial_metric_type = typename BenchmarkImpl::spatial_metric_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using det_spatial_metric_type =
      typename BenchmarkImpl::det_spatial_metric_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using christoffel_second_kind_type =
      typename BenchmarkImpl::christoffel_second_kind_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using normal_spacetime_vector_type =
      typename BenchmarkImpl::normal_spacetime_vector_type;
  using normal_spacetime_one_form_type =
      typename BenchmarkImpl::normal_spacetime_one_form_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;
  using d_spacetime_metric_type =
      typename BenchmarkImpl::d_spacetime_metric_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using spacetime_metric_type = typename BenchmarkImpl::spacetime_metric_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using phi_type = typename BenchmarkImpl::phi_type;
  using gamma0_type = typename BenchmarkImpl::gamma0_type;
  using gamma1_type = typename BenchmarkImpl::gamma1_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using spacetime_deriv_gauge_function_type =
      typename BenchmarkImpl::spacetime_deriv_gauge_function_type;
  // types not in Spectre implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using phi_one_normal_spatial_type =
      typename BenchmarkImpl::phi_one_normal_spatial_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: d_spacetime_metric
  const d_spacetime_metric_type d_spacetime_metric =
      make_with_random_values<d_spacetime_metric_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: d_pi
  const d_pi_type d_pi = make_with_random_values<d_pi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: d_phi
  const d_phi_type d_phi = make_with_random_values<d_phi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: spacetime_metric
  // In order to satisfy the physical requirements on the spacetime metric we
  // compute it from the helper functions that generate a physical lapse, shift,
  // and spatial metric.
  spacetime_metric_type spacetime_metric(used_for_size);
  gr::spacetime_metric(
      make_not_null(&spacetime_metric),
      TestHelpers::gr::random_lapse(generator, used_for_size),
      TestHelpers::gr::random_shift<Dim>(generator, used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(generator, used_for_size));

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: phi
  const phi_type phi = make_with_random_values<phi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: gamma0
  const gamma0_type gamma0 = make_with_random_values<gamma0_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: gamma1
  const gamma1_type gamma1 = make_with_random_values<gamma1_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: gamma2
  const gamma2_type gamma2 = make_with_random_values<gamma2_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: spacetime_deriv_gauge_function
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<spacetime_deriv_gauge_function_type>(
          generator, make_not_null(&distribution), used_for_size);

  // LHS: dt_spacetime_metric to be filled by manual impl
  dt_spacetime_metric_type dt_spacetime_metric_manual_filled(used_for_size);

  // LHS: dt_pi to be filled by manual impl
  dt_pi_type dt_pi_manual_filled(used_for_size);

  // LHS: dt_phi to be filled by manual impl
  dt_phi_type dt_phi_manual_filled(used_for_size);

  // LHS: temp_gamma1 to be filled by manual impl
  temp_gamma1_type temp_gamma1_manual_filled(used_for_size);

  // LHS: temp_gamma2 to be filled by manual impl
  temp_gamma2_type temp_gamma2_manual_filled(used_for_size);

  // LHS: gamma1gamma2 to be filled by manual impl
  gamma1gamma2_type gamma1gamma2_manual_filled(used_for_size);

  // LHS: pi_two_normals to be filled by manual impl
  pi_two_normals_type pi_two_normals_manual_filled(used_for_size);

  // LHS: normal_dot_gauge_constraint to be filled by manual impl
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint_manual_filled(
      used_for_size);

  // LHS: gamma1_plus_1 to be filled by manual impl
  gamma1_plus_1_type gamma1_plus_1_manual_filled(used_for_size);

  // LHS: pi_one_normal to be filled by manual impl
  pi_one_normal_type pi_one_normal_manual_filled(used_for_size);

  // LHS: gauge_constraint to be filled by manual impl
  gauge_constraint_type gauge_constraint_manual_filled(used_for_size);

  // LHS: phi_two_normals to be filled by manual impl
  phi_two_normals_type phi_two_normals_manual_filled(used_for_size);

  // LHS: shift_dot_three_index_constraint to be filled by manual impl
  shift_dot_three_index_constraint_type
      shift_dot_three_index_constraint_manual_filled(used_for_size);

  // LHS: phi_one_normal to be filled by manual impl
  phi_one_normal_type phi_one_normal_manual_filled(used_for_size);

  // LHS: pi_2_up to be filled by manual impl
  pi_2_up_type pi_2_up_manual_filled(used_for_size);

  // LHS: three_index_constraint to be filled by manual impl
  three_index_constraint_type three_index_constraint_manual_filled(
      used_for_size);

  // LHS: phi_1_up to be filled by manual impl
  phi_1_up_type phi_1_up_manual_filled(used_for_size);

  // LHS: phi_3_up to be filled by manual impl
  phi_3_up_type phi_3_up_manual_filled(used_for_size);

  // LHS: christoffel_first_kind_3_up to be filled by manual impl
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up_manual_filled(
      used_for_size);

  // LHS: lapse to be filled by manual impl
  lapse_type lapse_manual_filled(used_for_size);

  // LHS: shift to be filled by manual impl
  shift_type shift_manual_filled(used_for_size);

  // LHS: spatial_metric to be filled by manual impl
  spatial_metric_type spatial_metric_manual_filled(used_for_size);

  // LHS: inverse_spatial_metric to be filled by manual impl
  inverse_spatial_metric_type inverse_spatial_metric_manual_filled(
      used_for_size);

  // LHS: det_spatial_metric to be filled by manual impl
  det_spatial_metric_type det_spatial_metric_manual_filled(used_for_size);

  // LHS: inverse_spacetime_metric to be filled by manual impl
  inverse_spacetime_metric_type inverse_spacetime_metric_manual_filled(
      used_for_size);

  // LHS: christoffel_first_kind to be filled by manual impl
  christoffel_first_kind_type christoffel_first_kind_manual_filled(
      used_for_size);

  // LHS: christoffel_second_kind to be filled by manual impl
  christoffel_second_kind_type christoffel_second_kind_manual_filled(
      used_for_size);

  // LHS: trace_christoffel to be filled by manual impl
  trace_christoffel_type trace_christoffel_manual_filled(used_for_size);

  // LHS: normal_spacetime_vector to be filled by manual impl
  normal_spacetime_vector_type normal_spacetime_vector_manual_filled(
      used_for_size);

  // LHS: normal_spacetime_one_form to be filled by manual impl
  normal_spacetime_one_form_type normal_spacetime_one_form_manual_filled(
      used_for_size);

  // LHS: da_spacetime_metric to be filled by manual impl
  da_spacetime_metric_type da_spacetime_metric_manual_filled(used_for_size);

  // TEs can't iterate over only spatial indices of a spacetime index yet, so
  // where this is needed for the dt_pi and dt_phi calculations, these tensors
  // will be used, which hold only the spatial components to enable writing the
  // equations as closely as possible to how they appear in the manual loops

  // LHS: pi_one_normal_spatial to be filled by manual impl
  pi_one_normal_spatial_type pi_one_normal_spatial_manual_filled(used_for_size);

  // LHS: phi_one_normal_spatial to be filled by manual impl
  phi_one_normal_spatial_type phi_one_normal_spatial_manual_filled(
      used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(
      make_not_null(&dt_spacetime_metric_manual_filled),
      make_not_null(&dt_pi_manual_filled), make_not_null(&dt_phi_manual_filled),
      make_not_null(&temp_gamma1_manual_filled),
      make_not_null(&temp_gamma2_manual_filled),
      make_not_null(&gamma1gamma2_manual_filled),
      make_not_null(&pi_two_normals_manual_filled),
      make_not_null(&normal_dot_gauge_constraint_manual_filled),
      make_not_null(&gamma1_plus_1_manual_filled),
      make_not_null(&pi_one_normal_manual_filled),
      make_not_null(&gauge_constraint_manual_filled),
      make_not_null(&phi_two_normals_manual_filled),
      make_not_null(&shift_dot_three_index_constraint_manual_filled),
      make_not_null(&phi_one_normal_manual_filled),
      make_not_null(&pi_2_up_manual_filled),
      make_not_null(&three_index_constraint_manual_filled),
      make_not_null(&phi_1_up_manual_filled),
      make_not_null(&phi_3_up_manual_filled),
      make_not_null(&christoffel_first_kind_3_up_manual_filled),
      make_not_null(&lapse_manual_filled), make_not_null(&shift_manual_filled),
      make_not_null(&spatial_metric_manual_filled),
      make_not_null(&inverse_spatial_metric_manual_filled),
      make_not_null(&det_spatial_metric_manual_filled),
      make_not_null(&inverse_spacetime_metric_manual_filled),
      make_not_null(&christoffel_first_kind_manual_filled),
      make_not_null(&christoffel_second_kind_manual_filled),
      make_not_null(&trace_christoffel_manual_filled),
      make_not_null(&normal_spacetime_vector_manual_filled),
      make_not_null(&normal_spacetime_one_form_manual_filled),
      make_not_null(&da_spacetime_metric_manual_filled), d_spacetime_metric,
      d_pi, d_phi, spacetime_metric, pi, phi, gamma0, gamma1, gamma2,
      gauge_function, spacetime_deriv_gauge_function,
      make_not_null(&pi_one_normal_spatial_manual_filled),
      make_not_null(&phi_one_normal_spatial_manual_filled));

  // LHS: dt_spacetime_metric to be filled by TensorExpression impl<1>
  dt_spacetime_metric_type dt_spacetime_metric_te1_filled(used_for_size);

  // LHS: dt_pi to be filled by TensorExpression impl<1>
  dt_pi_type dt_pi_te1_filled(used_for_size);

  // LHS: dt_phi to be filled by TensorExpression impl<1>
  dt_phi_type dt_phi_te1_filled(used_for_size);

  // LHS: temp_gamma1 to be filled by TensorExpression impl<1>
  temp_gamma1_type temp_gamma1_te1_filled(used_for_size);

  // LHS: temp_gamma2 to be filled by TensorExpression impl<1>
  temp_gamma2_type temp_gamma2_te1_filled(used_for_size);

  // LHS: gamma1gamma2 to be filled by TensorExpression impl<1>
  gamma1gamma2_type gamma1gamma2_te1_filled(used_for_size);

  // LHS: pi_two_normals to be filled by TensorExpression impl<1>
  pi_two_normals_type pi_two_normals_te1_filled(used_for_size);

  // LHS: normal_dot_gauge_constraint to be filled by TensorExpression impl<1>
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint_te1_filled(
      used_for_size);

  // LHS: gamma1_plus_1 to be filled by TensorExpression impl<1>
  gamma1_plus_1_type gamma1_plus_1_te1_filled(used_for_size);

  // LHS: pi_one_normal to be filled by TensorExpression impl<1>
  pi_one_normal_type pi_one_normal_te1_filled(used_for_size);

  // LHS: gauge_constraint to be filled by TensorExpression impl<1>
  gauge_constraint_type gauge_constraint_te1_filled(used_for_size);

  // LHS: phi_two_normals to be filled by TensorExpression impl<1>
  phi_two_normals_type phi_two_normals_te1_filled(used_for_size);

  // LHS: shift_dot_three_index_constraint to be filled by TensorExpression
  // impl<1>
  shift_dot_three_index_constraint_type
      shift_dot_three_index_constraint_te1_filled(used_for_size);

  // LHS: phi_one_normal to be filled by TensorExpression impl<1>
  phi_one_normal_type phi_one_normal_te1_filled(used_for_size);

  // LHS: pi_2_up to be filled by TensorExpression impl<1>
  pi_2_up_type pi_2_up_te1_filled(used_for_size);

  // LHS: three_index_constraint to be filled by TensorExpression impl<1>
  three_index_constraint_type three_index_constraint_te1_filled(used_for_size);

  // LHS: phi_1_up to be filled by TensorExpression impl<1>
  phi_1_up_type phi_1_up_te1_filled(used_for_size);

  // LHS: phi_3_up to be filled by TensorExpression impl<1>
  phi_3_up_type phi_3_up_te1_filled(used_for_size);

  // LHS: christoffel_first_kind_3_up to be filled by TensorExpression impl<1>
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up_te1_filled(
      used_for_size);

  // LHS: lapse to be filled by TensorExpression impl<1>
  lapse_type lapse_te1_filled(used_for_size);

  // LHS: shift to be filled by TensorExpression impl<1>
  shift_type shift_te1_filled(used_for_size);

  // LHS: spatial_metric to be filled by TensorExpression impl<1>
  spatial_metric_type spatial_metric_te1_filled(used_for_size);

  // LHS: inverse_spatial_metric to be filled by TensorExpression impl<1>
  inverse_spatial_metric_type inverse_spatial_metric_te1_filled(used_for_size);

  // LHS: det_spatial_metric to be filled by TensorExpression impl<1>
  det_spatial_metric_type det_spatial_metric_te1_filled(used_for_size);

  // LHS: inverse_spacetime_metric to be filled by TensorExpression impl<1>
  inverse_spacetime_metric_type inverse_spacetime_metric_te1_filled(
      used_for_size);

  // LHS: christoffel_first_kind to be filled by TensorExpression impl<1>
  christoffel_first_kind_type christoffel_first_kind_te1_filled(used_for_size);

  // LHS: christoffel_second_kind to be filled by TensorExpression impl<1>
  christoffel_second_kind_type christoffel_second_kind_te1_filled(
      used_for_size);

  // LHS: trace_christoffel to be filled by TensorExpression impl<1>
  trace_christoffel_type trace_christoffel_te1_filled(used_for_size);

  // LHS: normal_spacetime_vector to be filled by TensorExpression impl<1>
  normal_spacetime_vector_type normal_spacetime_vector_te1_filled(
      used_for_size);

  // LHS: normal_spacetime_one_form to be filled by TensorExpression impl<1>
  normal_spacetime_one_form_type normal_spacetime_one_form_te1_filled(
      used_for_size);

  // LHS: da_spacetime_metric to be filled by TensorExpression impl<1>
  da_spacetime_metric_type da_spacetime_metric_te1_filled(used_for_size);

  // TEs can't iterate over only spatial indices of a spacetime index yet, so
  // where this is needed for the dt_pi and dt_phi calculations, these tensors
  // will be used, which hold only the spatial components to enable writing the
  // equations as closely as possible to how they appear in the manual loops

  // LHS: pi_one_normal_spatial to be filled by TensorExpression impl<1>
  pi_one_normal_spatial_type pi_one_normal_spatial_te1_filled(used_for_size);

  // LHS: phi_one_normal_spatial to be filled by TensorExpression impl<1>
  phi_one_normal_spatial_type phi_one_normal_spatial_te1_filled(used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&dt_spacetime_metric_te1_filled),
      make_not_null(&dt_pi_te1_filled), make_not_null(&dt_phi_te1_filled),
      make_not_null(&temp_gamma1_te1_filled),
      make_not_null(&temp_gamma2_te1_filled),
      make_not_null(&gamma1gamma2_te1_filled),
      make_not_null(&pi_two_normals_te1_filled),
      make_not_null(&normal_dot_gauge_constraint_te1_filled),
      make_not_null(&gamma1_plus_1_te1_filled),
      make_not_null(&pi_one_normal_te1_filled),
      make_not_null(&gauge_constraint_te1_filled),
      make_not_null(&phi_two_normals_te1_filled),
      make_not_null(&shift_dot_three_index_constraint_te1_filled),
      make_not_null(&phi_one_normal_te1_filled),
      make_not_null(&pi_2_up_te1_filled),
      make_not_null(&three_index_constraint_te1_filled),
      make_not_null(&phi_1_up_te1_filled), make_not_null(&phi_3_up_te1_filled),
      make_not_null(&christoffel_first_kind_3_up_te1_filled),
      make_not_null(&lapse_te1_filled), make_not_null(&shift_te1_filled),
      make_not_null(&spatial_metric_te1_filled),
      make_not_null(&inverse_spatial_metric_te1_filled),
      make_not_null(&det_spatial_metric_te1_filled),
      make_not_null(&inverse_spacetime_metric_te1_filled),
      make_not_null(&christoffel_first_kind_te1_filled),
      make_not_null(&christoffel_second_kind_te1_filled),
      make_not_null(&trace_christoffel_te1_filled),
      make_not_null(&normal_spacetime_vector_te1_filled),
      make_not_null(&normal_spacetime_one_form_te1_filled),
      make_not_null(&da_spacetime_metric_te1_filled), d_spacetime_metric, d_pi,
      d_phi, spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
      spacetime_deriv_gauge_function,
      make_not_null(&pi_one_normal_spatial_te1_filled),
      make_not_null(&phi_one_normal_spatial_te1_filled));

  // CHECK christoffel_first_kind (abb)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(christoffel_first_kind_manual_filled.get(a, b, c),
                              christoffel_first_kind_te1_filled.get(a, b, c));
      }
    }
  }

  // CHECK christoffel_second_kind (Abb)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te1_filled.get(a, b, c));
      }
    }
  }

  // CHECK trace_christoffel (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(trace_christoffel_manual_filled.get(a),
                          trace_christoffel_te1_filled.get(a));
  }

  // CHECK gamma1gamma2 (scalar)
  CHECK_ITERABLE_APPROX(gamma1gamma2_manual_filled.get(),
                        gamma1gamma2_te1_filled.get());

  // CHECK phi_1_up (Iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te1_filled.get(i, a, b));
      }
    }
  }

  // CHECK phi_3_up (iaB)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_3_up_manual_filled.get(i, a, b),
                              phi_3_up_te1_filled.get(i, a, b));
      }
    }
  }

  // CHECK pi_2_up (aB)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(pi_2_up_manual_filled.get(a, b),
                            pi_2_up_te1_filled.get(a, b));
    }
  }

  // CHECK christoffel_first_kind_3_up (abC)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(
            christoffel_first_kind_3_up_manual_filled.get(a, b, c),
            christoffel_first_kind_3_up_te1_filled.get(a, b, c));
      }
    }
  }

  // CHECK pi_one_normal (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_filled.get(a),
                          pi_one_normal_te1_filled.get(a));
  }

  // CHECK pi_one_normal_spatial (i)
  for (size_t i = 0; i < Dim; i++) {
    CHECK_ITERABLE_APPROX(pi_one_normal_spatial_manual_filled.get(i),
                          pi_one_normal_spatial_te1_filled.get(i));
  }

  // CHECK pi_two_normals (scalar)
  CHECK_ITERABLE_APPROX(pi_two_normals_manual_filled.get(),
                        pi_two_normals_te1_filled.get());

  // CHECK phi_one_normal (ia)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      CHECK_ITERABLE_APPROX(phi_one_normal_manual_filled.get(i, a),
                            phi_one_normal_te1_filled.get(i, a));
    }
  }

  // CHECK phi_one_normal_spatial (ij)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      CHECK_ITERABLE_APPROX(phi_one_normal_spatial_manual_filled.get(i, j),
                            phi_one_normal_spatial_te1_filled.get(i, j));
    }
  }

  // CHECK phi_two_normals (i)
  for (size_t i = 0; i < Dim; i++) {
    CHECK_ITERABLE_APPROX(phi_two_normals_manual_filled.get(i),
                          phi_two_normals_te1_filled.get(i));
  }

  // CHECK three_index_constraint (iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(three_index_constraint_manual_filled.get(i, a, b),
                              three_index_constraint_te1_filled.get(i, a, b));
      }
    }
  }

  // CHECK gauge_constraint (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(gauge_constraint_manual_filled.get(a),
                          gauge_constraint_te1_filled.get(a));
  }

  // CHECK normal_dot_gauge_constraint (scalar)
  CHECK_ITERABLE_APPROX(normal_dot_gauge_constraint_manual_filled.get(),
                        normal_dot_gauge_constraint_te1_filled.get());

  // CHECK gamma1_plus_1 (scalar)
  CHECK_ITERABLE_APPROX(gamma1_plus_1_manual_filled.get(),
                        gamma1_plus_1_te1_filled.get());

  // CHECK shift_dot_three_index_constraint (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(
          shift_dot_three_index_constraint_manual_filled.get(a, b),
          shift_dot_three_index_constraint_te1_filled.get(a, b));
    }
  }

  // CHECK dt_spacetime_metric (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                            dt_spacetime_metric_te1_filled.get(a, b));
    }
  }

  // CHECK dt_pi (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_pi_manual_filled.get(a, b),
                            dt_pi_te1_filled.get(a, b));
    }
  }

  // CHECK dt_phi (iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(dt_phi_manual_filled.get(i, a, b),
                              dt_phi_te1_filled.get(i, a, b));
      }
    }
  }

  // === Check TE impl with TempTensors ===

  size_t num_grid_points = 0;
  if constexpr (std::is_same_v<DataType, DataVector>) {
    num_grid_points = used_for_size.size();
  }

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_spacetime_metric_type>,
      ::Tags::TempTensor<1, dt_pi_type>, ::Tags::TempTensor<2, dt_phi_type>,
      ::Tags::TempTensor<3, temp_gamma1_type>,
      ::Tags::TempTensor<4, temp_gamma2_type>,
      ::Tags::TempTensor<5, gamma1gamma2_type>,
      ::Tags::TempTensor<6, pi_two_normals_type>,
      ::Tags::TempTensor<7, normal_dot_gauge_constraint_type>,
      ::Tags::TempTensor<8, gamma1_plus_1_type>,
      ::Tags::TempTensor<9, pi_one_normal_type>,
      ::Tags::TempTensor<10, gauge_constraint_type>,
      ::Tags::TempTensor<11, phi_two_normals_type>,
      ::Tags::TempTensor<12, shift_dot_three_index_constraint_type>,
      ::Tags::TempTensor<13, phi_one_normal_type>,
      ::Tags::TempTensor<14, pi_2_up_type>,
      ::Tags::TempTensor<15, three_index_constraint_type>,
      ::Tags::TempTensor<16, phi_1_up_type>,
      ::Tags::TempTensor<17, phi_3_up_type>,
      ::Tags::TempTensor<18, christoffel_first_kind_3_up_type>,
      ::Tags::TempTensor<19, lapse_type>, ::Tags::TempTensor<20, shift_type>,
      ::Tags::TempTensor<21, spatial_metric_type>,
      ::Tags::TempTensor<22, inverse_spatial_metric_type>,
      ::Tags::TempTensor<23, det_spatial_metric_type>,
      ::Tags::TempTensor<24, inverse_spacetime_metric_type>,
      ::Tags::TempTensor<25, christoffel_first_kind_type>,
      ::Tags::TempTensor<26, christoffel_second_kind_type>,
      ::Tags::TempTensor<27, trace_christoffel_type>,
      ::Tags::TempTensor<28, normal_spacetime_vector_type>,
      ::Tags::TempTensor<29, normal_spacetime_one_form_type>,
      ::Tags::TempTensor<30, da_spacetime_metric_type>,
      ::Tags::TempTensor<31, d_spacetime_metric_type>,
      ::Tags::TempTensor<32, d_pi_type>, ::Tags::TempTensor<33, d_phi_type>,
      ::Tags::TempTensor<34, spacetime_metric_type>,
      ::Tags::TempTensor<35, pi_type>, ::Tags::TempTensor<36, phi_type>,
      ::Tags::TempTensor<37, gamma0_type>, ::Tags::TempTensor<38, gamma1_type>,
      ::Tags::TempTensor<39, gamma2_type>,
      ::Tags::TempTensor<40, gauge_function_type>,
      ::Tags::TempTensor<41, spacetime_deriv_gauge_function_type>,
      ::Tags::TempTensor<42, pi_one_normal_spatial_type>,
      ::Tags::TempTensor<43, phi_one_normal_spatial_type>>>
      vars{num_grid_points};

  // RHS: d_spacetime_metric
  d_spacetime_metric_type& d_spacetime_metric_te_temp =
      get<::Tags::TempTensor<31, d_spacetime_metric_type>>(vars);
  copy_tensor(d_spacetime_metric, make_not_null(&d_spacetime_metric_te_temp));

  // RHS: d_pi
  d_pi_type& d_pi_te_temp = get<::Tags::TempTensor<32, d_pi_type>>(vars);
  copy_tensor(d_pi, make_not_null(&d_pi_te_temp));

  // RHS: d_phi
  d_phi_type& d_phi_te_temp = get<::Tags::TempTensor<33, d_phi_type>>(vars);
  copy_tensor(d_phi, make_not_null(&d_phi_te_temp));

  // RHS: spacetime_metric
  spacetime_metric_type& spacetime_metric_te_temp =
      get<::Tags::TempTensor<34, spacetime_metric_type>>(vars);
  copy_tensor(spacetime_metric, make_not_null(&spacetime_metric_te_temp));

  // RHS: pi
  pi_type& pi_te_temp = get<::Tags::TempTensor<35, pi_type>>(vars);
  copy_tensor(pi, make_not_null(&pi_te_temp));

  // RHS: phi
  phi_type& phi_te_temp = get<::Tags::TempTensor<36, phi_type>>(vars);
  copy_tensor(phi, make_not_null(&phi_te_temp));

  // RHS: gamma0
  gamma0_type& gamma0_te_temp = get<::Tags::TempTensor<37, gamma0_type>>(vars);
  copy_tensor(gamma0, make_not_null(&gamma0_te_temp));

  // RHS: gamma1
  gamma1_type& gamma1_te_temp = get<::Tags::TempTensor<38, gamma1_type>>(vars);
  copy_tensor(gamma1, make_not_null(&gamma1_te_temp));

  // RHS: gamma2
  gamma2_type& gamma2_te_temp = get<::Tags::TempTensor<39, gamma2_type>>(vars);
  copy_tensor(gamma2, make_not_null(&gamma2_te_temp));

  // RHS: gauge_function
  gauge_function_type& gauge_function_te_temp =
      get<::Tags::TempTensor<40, gauge_function_type>>(vars);
  copy_tensor(gauge_function, make_not_null(&gauge_function_te_temp));

  // RHS: spacetime_deriv_gauge_function
  spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function_te_temp =
      get<::Tags::TempTensor<41, spacetime_deriv_gauge_function_type>>(vars);
  copy_tensor(spacetime_deriv_gauge_function,
              make_not_null(&spacetime_deriv_gauge_function_te_temp));

  // LHS: dt_spacetime_metric impl<1>
  dt_spacetime_metric_type& dt_spacetime_metric_te1_temp =
      get<::Tags::TempTensor<0, dt_spacetime_metric_type>>(vars);

  // LHS: dt_pi impl<1>
  dt_pi_type& dt_pi_te1_temp = get<::Tags::TempTensor<1, dt_pi_type>>(vars);

  // LHS: dt_phi impl<1>
  dt_phi_type& dt_phi_te1_temp = get<::Tags::TempTensor<2, dt_phi_type>>(vars);

  // LHS: temp_gamma1 impl<1>
  temp_gamma1_type& temp_gamma1_te1_temp =
      get<::Tags::TempTensor<3, temp_gamma1_type>>(vars);

  // LHS: temp_gamma2 impl<1>
  temp_gamma2_type& temp_gamma2_te1_temp =
      get<::Tags::TempTensor<4, temp_gamma2_type>>(vars);

  // LHS: gamma1gamma2 impl<1>
  gamma1gamma2_type& gamma1gamma2_te1_temp =
      get<::Tags::TempTensor<5, gamma1gamma2_type>>(vars);

  // LHS: pi_two_normals impl<1>
  pi_two_normals_type& pi_two_normals_te1_temp =
      get<::Tags::TempTensor<6, pi_two_normals_type>>(vars);

  // LHS: normal_dot_gauge_constraint impl<1>
  normal_dot_gauge_constraint_type& normal_dot_gauge_constraint_te1_temp =
      get<::Tags::TempTensor<7, normal_dot_gauge_constraint_type>>(vars);

  // LHS: gamma1_plus_1 impl<1>
  gamma1_plus_1_type& gamma1_plus_1_te1_temp =
      get<::Tags::TempTensor<8, gamma1_plus_1_type>>(vars);

  // LHS: pi_one_normal impl<1>
  pi_one_normal_type& pi_one_normal_te1_temp =
      get<::Tags::TempTensor<9, pi_one_normal_type>>(vars);

  // LHS: gauge_constraint impl<1>
  gauge_constraint_type& gauge_constraint_te1_temp =
      get<::Tags::TempTensor<10, gauge_constraint_type>>(vars);

  // LHS: phi_two_normals impl<1>
  phi_two_normals_type& phi_two_normals_te1_temp =
      get<::Tags::TempTensor<11, phi_two_normals_type>>(vars);

  // LHS: shift_dot_three_index_constraint impl<1>
  shift_dot_three_index_constraint_type&
      shift_dot_three_index_constraint_te1_temp =
          get<::Tags::TempTensor<12, shift_dot_three_index_constraint_type>>(
              vars);

  // LHS: phi_one_normal impl<1>
  phi_one_normal_type& phi_one_normal_te1_temp =
      get<::Tags::TempTensor<13, phi_one_normal_type>>(vars);

  // LHS: pi_2_up impl<1>
  pi_2_up_type& pi_2_up_te1_temp =
      get<::Tags::TempTensor<14, pi_2_up_type>>(vars);

  // LHS: three_index_constraint impl<1>
  three_index_constraint_type& three_index_constraint_te1_temp =
      get<::Tags::TempTensor<15, three_index_constraint_type>>(vars);

  // LHS: phi_1_up impl<1>
  phi_1_up_type& phi_1_up_te1_temp =
      get<::Tags::TempTensor<16, phi_1_up_type>>(vars);

  // LHS: phi_3_up impl<1>
  phi_3_up_type& phi_3_up_te1_temp =
      get<::Tags::TempTensor<17, phi_3_up_type>>(vars);

  // LHS: christoffel_first_kind_3_up impl<1>
  christoffel_first_kind_3_up_type& christoffel_first_kind_3_up_te1_temp =
      get<::Tags::TempTensor<18, christoffel_first_kind_3_up_type>>(vars);

  // LHS: lapse impl<1>
  lapse_type& lapse_te1_temp = get<::Tags::TempTensor<19, lapse_type>>(vars);

  // LHS: shift impl<1>
  shift_type& shift_te1_temp = get<::Tags::TempTensor<20, shift_type>>(vars);

  // LHS: spatial_metric impl<1>
  spatial_metric_type& spatial_metric_te1_temp =
      get<::Tags::TempTensor<21, spatial_metric_type>>(vars);

  // LHS: inverse_spatial_metric impl<1>
  inverse_spatial_metric_type& inverse_spatial_metric_te1_temp =
      get<::Tags::TempTensor<22, inverse_spatial_metric_type>>(vars);

  // LHS: det_spatial_metric impl<1>
  det_spatial_metric_type& det_spatial_metric_te1_temp =
      get<::Tags::TempTensor<23, det_spatial_metric_type>>(vars);

  // LHS: inverse_spacetime_metric impl<1>
  inverse_spacetime_metric_type& inverse_spacetime_metric_te1_temp =
      get<::Tags::TempTensor<24, inverse_spacetime_metric_type>>(vars);

  // LHS: christoffel_first_kind impl<1>
  christoffel_first_kind_type& christoffel_first_kind_te1_temp =
      get<::Tags::TempTensor<25, christoffel_first_kind_type>>(vars);

  // LHS: christoffel_second_kind impl<1>
  christoffel_second_kind_type& christoffel_second_kind_te1_temp =
      get<::Tags::TempTensor<26, christoffel_second_kind_type>>(vars);

  // LHS: trace_christoffel impl<1>
  trace_christoffel_type& trace_christoffel_te1_temp =
      get<::Tags::TempTensor<27, trace_christoffel_type>>(vars);

  // LHS: normal_spacetime_vector impl<1>
  normal_spacetime_vector_type& normal_spacetime_vector_te1_temp =
      get<::Tags::TempTensor<28, normal_spacetime_vector_type>>(vars);

  // LHS: normal_spacetime_one_form impl<1>
  normal_spacetime_one_form_type& normal_spacetime_one_form_te1_temp =
      get<::Tags::TempTensor<29, normal_spacetime_one_form_type>>(vars);

  // LHS: da_spacetime_metric impl<1>
  da_spacetime_metric_type& da_spacetime_metric_te1_temp =
      get<::Tags::TempTensor<30, da_spacetime_metric_type>>(vars);

  // LHS: pi_one_normal_spatial impl<1>
  pi_one_normal_spatial_type& pi_one_normal_spatial_te1_temp =
      get<::Tags::TempTensor<42, pi_one_normal_spatial_type>>(vars);

  // LHS: phi_one_normal_spatial impl<1>
  phi_one_normal_spatial_type& phi_one_normal_spatial_te1_temp =
      get<::Tags::TempTensor<43, phi_one_normal_spatial_type>>(vars);

  // Compute TensorExpression impl<1> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&dt_spacetime_metric_te1_temp),
      make_not_null(&dt_pi_te1_temp), make_not_null(&dt_phi_te1_temp),
      make_not_null(&temp_gamma1_te1_temp),
      make_not_null(&temp_gamma2_te1_temp),
      make_not_null(&gamma1gamma2_te1_temp),
      make_not_null(&pi_two_normals_te1_temp),
      make_not_null(&normal_dot_gauge_constraint_te1_temp),
      make_not_null(&gamma1_plus_1_te1_temp),
      make_not_null(&pi_one_normal_te1_temp),
      make_not_null(&gauge_constraint_te1_temp),
      make_not_null(&phi_two_normals_te1_temp),
      make_not_null(&shift_dot_three_index_constraint_te1_temp),
      make_not_null(&phi_one_normal_te1_temp), make_not_null(&pi_2_up_te1_temp),
      make_not_null(&three_index_constraint_te1_temp),
      make_not_null(&phi_1_up_te1_temp), make_not_null(&phi_3_up_te1_temp),
      make_not_null(&christoffel_first_kind_3_up_te1_temp),
      make_not_null(&lapse_te1_temp), make_not_null(&shift_te1_temp),
      make_not_null(&spatial_metric_te1_temp),
      make_not_null(&inverse_spatial_metric_te1_temp),
      make_not_null(&det_spatial_metric_te1_temp),
      make_not_null(&inverse_spacetime_metric_te1_temp),
      make_not_null(&christoffel_first_kind_te1_temp),
      make_not_null(&christoffel_second_kind_te1_temp),
      make_not_null(&trace_christoffel_te1_temp),
      make_not_null(&normal_spacetime_vector_te1_temp),
      make_not_null(&normal_spacetime_one_form_te1_temp),
      make_not_null(&da_spacetime_metric_te1_temp), d_spacetime_metric_te_temp,
      d_pi_te_temp, d_phi_te_temp, spacetime_metric_te_temp, pi_te_temp,
      phi_te_temp, gamma0_te_temp, gamma1_te_temp, gamma2_te_temp,
      gauge_function_te_temp, spacetime_deriv_gauge_function_te_temp,
      make_not_null(&pi_one_normal_spatial_te1_temp),
      make_not_null(&phi_one_normal_spatial_te1_temp));

  // CHECK christoffel_first_kind (abb)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(christoffel_first_kind_manual_filled.get(a, b, c),
                              christoffel_first_kind_te1_temp.get(a, b, c));
      }
    }
  }

  // CHECK christoffel_second_kind (Abb)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te1_temp.get(a, b, c));
      }
    }
  }

  // CHECK trace_christoffel (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(trace_christoffel_manual_filled.get(a),
                          trace_christoffel_te1_temp.get(a));
  }

  // CHECK gamma1gamma2 (scalar)
  CHECK_ITERABLE_APPROX(gamma1gamma2_manual_filled.get(),
                        gamma1gamma2_te1_temp.get());

  // CHECK phi_1_up (Iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te1_temp.get(i, a, b));
      }
    }
  }

  // CHECK phi_3_up (iaB)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_3_up_manual_filled.get(i, a, b),
                              phi_3_up_te1_temp.get(i, a, b));
      }
    }
  }

  // CHECK pi_2_up (aB)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(pi_2_up_manual_filled.get(a, b),
                            pi_2_up_te1_temp.get(a, b));
    }
  }

  // CHECK christoffel_first_kind_3_up (abC)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(
            christoffel_first_kind_3_up_manual_filled.get(a, b, c),
            christoffel_first_kind_3_up_te1_temp.get(a, b, c));
      }
    }
  }

  // CHECK pi_one_normal (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_filled.get(a),
                          pi_one_normal_te1_temp.get(a));
  }

  // CHECK pi_one_normal_spatial (i)
  for (size_t i = 0; i < Dim; i++) {
    CHECK_ITERABLE_APPROX(pi_one_normal_spatial_manual_filled.get(i),
                          pi_one_normal_spatial_te1_temp.get(i));
  }

  // CHECK pi_two_normals (scalar)
  CHECK_ITERABLE_APPROX(pi_two_normals_manual_filled.get(),
                        pi_two_normals_te1_temp.get());

  // CHECK phi_one_normal (ia)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      CHECK_ITERABLE_APPROX(phi_one_normal_manual_filled.get(i, a),
                            phi_one_normal_te1_temp.get(i, a));
    }
  }

  // CHECK phi_one_normal_spatial (ij)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      CHECK_ITERABLE_APPROX(phi_one_normal_spatial_manual_filled.get(i, j),
                            phi_one_normal_spatial_te1_temp.get(i, j));
    }
  }

  // CHECK phi_two_normals (i)
  for (size_t i = 0; i < Dim; i++) {
    CHECK_ITERABLE_APPROX(phi_two_normals_manual_filled.get(i),
                          phi_two_normals_te1_temp.get(i));
  }

  // CHECK three_index_constraint (iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(three_index_constraint_manual_filled.get(i, a, b),
                              three_index_constraint_te1_temp.get(i, a, b));
      }
    }
  }

  // CHECK gauge_constraint (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(gauge_constraint_manual_filled.get(a),
                          gauge_constraint_te1_temp.get(a));
  }

  // CHECK normal_dot_gauge_constraint (scalar)
  CHECK_ITERABLE_APPROX(normal_dot_gauge_constraint_manual_filled.get(),
                        normal_dot_gauge_constraint_te1_temp.get());

  // CHECK gamma1_plus_1 (scalar)
  CHECK_ITERABLE_APPROX(gamma1_plus_1_manual_filled.get(),
                        gamma1_plus_1_te1_temp.get());

  // CHECK shift_dot_three_index_constraint (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(
          shift_dot_three_index_constraint_manual_filled.get(a, b),
          shift_dot_three_index_constraint_te1_temp.get(a, b));
    }
  }

  // CHECK dt_spacetime_metric (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                            dt_spacetime_metric_te1_temp.get(a, b));
    }
  }

  // CHECK dt_pi (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_pi_manual_filled.get(a, b),
                            dt_pi_te1_temp.get(a, b));
    }
  }

  // CHECK dt_phi (iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(dt_phi_manual_filled.get(i, a, b),
                              dt_phi_te1_temp.get(i, a, b));
      }
    }
  }
}

template <typename DataType, typename Generator>
void test_benchmarked_impls(
    const DataType& used_for_size,
    const gsl::not_null<Generator*> generator) noexcept {
  test_benchmarked_impls_core<1>(used_for_size, generator);
  test_benchmarked_impls_core<2>(used_for_size, generator);
  test_benchmarked_impls_core<3>(used_for_size, generator);
}

SPECTRE_TEST_CASE("Unit.Benchmark.GHTimeDerivative.all",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_impls(std::numeric_limits<double>::signaling_NaN(),
                         make_not_null(&generator));
  test_benchmarked_impls(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
