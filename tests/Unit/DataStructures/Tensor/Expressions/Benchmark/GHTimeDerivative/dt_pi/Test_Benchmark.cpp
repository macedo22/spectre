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
#include "Executables/Benchmark/GHTimeDerivative/dt_pi/BenchmarkedImpls.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
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
  using dt_pi_type = typename BenchmarkImpl::dt_pi_type;
  using spacetime_deriv_gauge_function_type =
      typename BenchmarkImpl::spacetime_deriv_gauge_function_type;
  using pi_two_normals_type = typename BenchmarkImpl::pi_two_normals_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using gamma0_type = typename BenchmarkImpl::gamma0_type;
  using normal_spacetime_one_form_type =
      typename BenchmarkImpl::normal_spacetime_one_form_type;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using spacetime_metric_type = typename BenchmarkImpl::spacetime_metric_type;
  using normal_dot_gauge_constraint_type =
      typename BenchmarkImpl::normal_dot_gauge_constraint_type;
  using christoffel_second_kind_type =
      typename BenchmarkImpl::christoffel_second_kind_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using pi_2_up_type = typename BenchmarkImpl::pi_2_up_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using phi_3_up_type = typename BenchmarkImpl::phi_3_up_type;
  using christoffel_first_kind_3_up_type =
      typename BenchmarkImpl::christoffel_first_kind_3_up_type;
  // type not in SpECTRE implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: spacetime_deriv_gauge_function
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<spacetime_deriv_gauge_function_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: pi_two_normals
  const pi_two_normals_type pi_two_normals =
      make_with_random_values<pi_two_normals_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: gamma0
  const gamma0_type gamma0 = make_with_random_values<gamma0_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: normal_spacetime_one_form
  const normal_spacetime_one_form_type normal_spacetime_one_form =
      make_with_random_values<normal_spacetime_one_form_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: gauge_constraint
  const gauge_constraint_type gauge_constraint =
      make_with_random_values<gauge_constraint_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: spacetime_metric
  const spacetime_metric_type spacetime_metric =
      make_with_random_values<spacetime_metric_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: normal_dot_gauge_constraint
  const normal_dot_gauge_constraint_type normal_dot_gauge_constraint =
      make_with_random_values<normal_dot_gauge_constraint_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: christoffel_second_kind
  const christoffel_second_kind_type christoffel_second_kind =
      make_with_random_values<christoffel_second_kind_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: pi_2_up
  const pi_2_up_type pi_2_up = make_with_random_values<pi_2_up_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: phi_1_up
  const phi_1_up_type phi_1_up = make_with_random_values<phi_1_up_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: phi_3_up
  const phi_3_up_type phi_3_up = make_with_random_values<phi_3_up_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: christoffel_first_kind_3_up
  const christoffel_first_kind_3_up_type christoffel_first_kind_3_up =
      make_with_random_values<christoffel_first_kind_3_up_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: pi_one_normal_spatial
  const pi_one_normal_spatial_type pi_one_normal_spatial =
      make_with_random_values<pi_one_normal_spatial_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: inverse_spatial_metric
  const inverse_spatial_metric_type inverse_spatial_metric =
      make_with_random_values<inverse_spatial_metric_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: d_phi
  const d_phi_type d_phi = make_with_random_values<d_phi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: gamma1gamma2
  const gamma1gamma2_type gamma1gamma2 =
      make_with_random_values<gamma1gamma2_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: shift_dot_three_index_constraint
  const shift_dot_three_index_constraint_type shift_dot_three_index_constraint =
      make_with_random_values<shift_dot_three_index_constraint_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: shift
  const shift_type shift = make_with_random_values<shift_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: d_pi
  const d_pi_type d_pi = make_with_random_values<d_pi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // LHS: dt_pi to be filled by manual impl
  dt_pi_type dt_pi_manual_filled(used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(
      make_not_null(&dt_pi_manual_filled), spacetime_deriv_gauge_function,
      pi_two_normals, pi, gamma0, normal_spacetime_one_form, gauge_constraint,
      spacetime_metric, normal_dot_gauge_constraint, christoffel_second_kind,
      gauge_function, pi_2_up, phi_1_up, phi_3_up, christoffel_first_kind_3_up,
      pi_one_normal_spatial, inverse_spatial_metric, d_phi, lapse, gamma1gamma2,
      shift_dot_three_index_constraint, shift, d_pi);

  // LHS: dt_pi to be filled by TensorExpression impl<1>
  dt_pi_type dt_pi_te1_filled(used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&dt_pi_te1_filled), spacetime_deriv_gauge_function,
      pi_two_normals, pi, gamma0, normal_spacetime_one_form, gauge_constraint,
      spacetime_metric, normal_dot_gauge_constraint, christoffel_second_kind,
      gauge_function, pi_2_up, phi_1_up, phi_3_up, christoffel_first_kind_3_up,
      pi_one_normal_spatial, inverse_spatial_metric, d_phi, lapse, gamma1gamma2,
      shift_dot_three_index_constraint, shift, d_pi);

  // CHECK dt_pi
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_pi_manual_filled.get(a, b),
                            dt_pi_te1_filled.get(a, b));
    }
  }

  // === Check TE impl with TempTensors ===

  size_t num_grid_points = 0;
  if constexpr (std::is_same_v<DataType, DataVector>) {
    num_grid_points = used_for_size.size();
  }

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_pi_type>,
      ::Tags::TempTensor<1, spacetime_deriv_gauge_function_type>,
      ::Tags::TempTensor<2, pi_two_normals_type>,
      ::Tags::TempTensor<3, pi_type>, ::Tags::TempTensor<4, gamma0_type>,
      ::Tags::TempTensor<5, normal_spacetime_one_form_type>,
      ::Tags::TempTensor<6, gauge_constraint_type>,
      ::Tags::TempTensor<7, spacetime_metric_type>,
      ::Tags::TempTensor<8, normal_dot_gauge_constraint_type>,
      ::Tags::TempTensor<9, christoffel_second_kind_type>,
      ::Tags::TempTensor<10, gauge_function_type>,
      ::Tags::TempTensor<11, pi_2_up_type>,
      ::Tags::TempTensor<12, phi_1_up_type>,
      ::Tags::TempTensor<13, phi_3_up_type>,
      ::Tags::TempTensor<14, christoffel_first_kind_3_up_type>,
      ::Tags::TempTensor<15, pi_one_normal_spatial_type>,
      ::Tags::TempTensor<16, inverse_spatial_metric_type>,
      ::Tags::TempTensor<17, d_phi_type>, ::Tags::TempTensor<18, lapse_type>,
      ::Tags::TempTensor<19, gamma1gamma2_type>,
      ::Tags::TempTensor<20, shift_dot_three_index_constraint_type>,
      ::Tags::TempTensor<21, shift_type>, ::Tags::TempTensor<22, d_pi_type>>>
      vars{num_grid_points};

  // RHS: spacetime_deriv_gauge_function
  spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function_te_temp =
      get<::Tags::TempTensor<1, spacetime_deriv_gauge_function_type>>(vars);
  copy_tensor(spacetime_deriv_gauge_function,
              make_not_null(&spacetime_deriv_gauge_function_te_temp));

  // RHS: pi_two_normals
  pi_two_normals_type& pi_two_normals_te_temp =
      get<::Tags::TempTensor<2, pi_two_normals_type>>(vars);
  copy_tensor(pi_two_normals, make_not_null(&pi_two_normals_te_temp));

  // RHS: pi
  pi_type& pi_te_temp = get<::Tags::TempTensor<3, pi_type>>(vars);
  copy_tensor(pi, make_not_null(&pi_te_temp));

  // RHS: gamma0
  gamma0_type& gamma0_te_temp = get<::Tags::TempTensor<4, gamma0_type>>(vars);
  copy_tensor(gamma0, make_not_null(&gamma0_te_temp));

  // RHS: normal_spacetime_one_form
  normal_spacetime_one_form_type& normal_spacetime_one_form_te_temp =
      get<::Tags::TempTensor<5, normal_spacetime_one_form_type>>(vars);
  copy_tensor(normal_spacetime_one_form,
              make_not_null(&normal_spacetime_one_form_te_temp));

  // RHS: gauge_constraint
  gauge_constraint_type& gauge_constraint_te_temp =
      get<::Tags::TempTensor<6, gauge_constraint_type>>(vars);
  copy_tensor(gauge_constraint, make_not_null(&gauge_constraint_te_temp));

  // RHS: spacetime_metric
  spacetime_metric_type& spacetime_metric_te_temp =
      get<::Tags::TempTensor<7, spacetime_metric_type>>(vars);
  copy_tensor(spacetime_metric, make_not_null(&spacetime_metric_te_temp));

  // RHS: normal_dot_gauge_constraint
  normal_dot_gauge_constraint_type& normal_dot_gauge_constraint_te_temp =
      get<::Tags::TempTensor<8, normal_dot_gauge_constraint_type>>(vars);
  copy_tensor(normal_dot_gauge_constraint,
              make_not_null(&normal_dot_gauge_constraint_te_temp));

  // RHS: christoffel_second_kind
  christoffel_second_kind_type& christoffel_second_kind_te_temp =
      get<::Tags::TempTensor<9, christoffel_second_kind_type>>(vars);
  copy_tensor(christoffel_second_kind,
              make_not_null(&christoffel_second_kind_te_temp));

  // RHS: gauge_function
  gauge_function_type& gauge_function_te_temp =
      get<::Tags::TempTensor<10, gauge_function_type>>(vars);
  copy_tensor(gauge_function, make_not_null(&gauge_function_te_temp));

  // RHS: pi_2_up
  pi_2_up_type& pi_2_up_te_temp =
      get<::Tags::TempTensor<11, pi_2_up_type>>(vars);
  copy_tensor(pi_2_up, make_not_null(&pi_2_up_te_temp));

  // RHS: phi_1_up
  phi_1_up_type& phi_1_up_te_temp =
      get<::Tags::TempTensor<12, phi_1_up_type>>(vars);
  copy_tensor(phi_1_up, make_not_null(&phi_1_up_te_temp));

  // RHS: phi_3_up
  phi_3_up_type& phi_3_up_te_temp =
      get<::Tags::TempTensor<13, phi_3_up_type>>(vars);
  copy_tensor(phi_3_up, make_not_null(&phi_3_up_te_temp));

  // RHS: christoffel_first_kind_3_up
  christoffel_first_kind_3_up_type& christoffel_first_kind_3_up_te_temp =
      get<::Tags::TempTensor<14, christoffel_first_kind_3_up_type>>(vars);
  copy_tensor(christoffel_first_kind_3_up,
              make_not_null(&christoffel_first_kind_3_up_te_temp));

  // RHS: pi_one_normal_spatial
  pi_one_normal_spatial_type& pi_one_normal_spatial_te_temp =
      get<::Tags::TempTensor<15, pi_one_normal_spatial_type>>(vars);
  copy_tensor(pi_one_normal_spatial,
              make_not_null(&pi_one_normal_spatial_te_temp));

  // RHS: inverse_spatial_metric
  inverse_spatial_metric_type& inverse_spatial_metric_te_temp =
      get<::Tags::TempTensor<16, inverse_spatial_metric_type>>(vars);
  copy_tensor(inverse_spatial_metric,
              make_not_null(&inverse_spatial_metric_te_temp));

  // RHS: d_phi
  d_phi_type& d_phi_te_temp = get<::Tags::TempTensor<17, d_phi_type>>(vars);
  copy_tensor(d_phi, make_not_null(&d_phi_te_temp));

  // RHS: lapse
  lapse_type& lapse_te_temp = get<::Tags::TempTensor<18, lapse_type>>(vars);
  copy_tensor(lapse, make_not_null(&lapse_te_temp));

  // RHS: gamma1gamma2
  gamma1gamma2_type& gamma1gamma2_te_temp =
      get<::Tags::TempTensor<19, gamma1gamma2_type>>(vars);
  copy_tensor(gamma1gamma2, make_not_null(&lapse_te_temp));

  // RHS: shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type&
      shift_dot_three_index_constraint_te_temp =
          get<::Tags::TempTensor<20, shift_dot_three_index_constraint_type>>(
              vars);
  copy_tensor(shift_dot_three_index_constraint,
              make_not_null(&shift_dot_three_index_constraint_te_temp));

  // RHS: shift
  shift_type& shift_te_temp = get<::Tags::TempTensor<21, shift_type>>(vars);
  copy_tensor(shift, make_not_null(&shift_te_temp));

  // RHS: d_pi
  d_pi_type& d_pi_te_temp = get<::Tags::TempTensor<22, d_pi_type>>(vars);
  copy_tensor(d_pi, make_not_null(&d_pi_te_temp));

  // LHS: dt_pi impl<1>
  dt_pi_type& dt_pi_te1_temp = get<::Tags::TempTensor<0, dt_pi_type>>(vars);

  // Compute TensorExpression impl<1> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&dt_pi_te1_temp), spacetime_deriv_gauge_function_te_temp,
      pi_two_normals_te_temp, pi_te_temp, gamma0_te_temp,
      normal_spacetime_one_form_te_temp, gauge_constraint_te_temp,
      spacetime_metric_te_temp, normal_dot_gauge_constraint_te_temp,
      christoffel_second_kind_te_temp, gauge_function_te_temp, pi_2_up_te_temp,
      phi_1_up_te_temp, phi_3_up_te_temp, christoffel_first_kind_3_up_te_temp,
      pi_one_normal_spatial_te_temp, inverse_spatial_metric_te_temp,
      d_phi_te_temp, lapse_te_temp, gamma1gamma2_te_temp,
      shift_dot_three_index_constraint_te_temp, shift_te_temp, d_pi_te_temp);

  // CHECK dt_pi
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_pi_manual_filled.get(a, b),
                            dt_pi_te1_temp.get(a, b));
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

SPECTRE_TEST_CASE("Unit.Benchmark.GHTimeDerivative.dt_pi",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_impls(std::numeric_limits<double>::signaling_NaN(),
                         make_not_null(&generator));
  test_benchmarked_impls(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
