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
#include "Executables/Benchmark/GHTimeDerivative/dt_phi/BenchmarkedImpls.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename... Ts>
void copy_tensor(const Tensor<Ts...>& tensor_source,
                 gsl::not_null<Tensor<Ts...>*> tensor_destination) {
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
void test_benchmarked_impls_core(const DataType& used_for_size,
                                 const gsl::not_null<Generator*> generator) {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: d_pi
  const d_pi_type d_pi = make_with_random_values<d_pi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: phi_two_normals
  const phi_two_normals_type phi_two_normals =
      make_with_random_values<phi_two_normals_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: three_index_constraint
  const three_index_constraint_type three_index_constraint =
      make_with_random_values<three_index_constraint_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: gamma2
  const gamma2_type gamma2 = make_with_random_values<gamma2_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: phi_one_normal
  const phi_one_normal_type phi_one_normal =
      make_with_random_values<phi_one_normal_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: phi_1_up
  const phi_1_up_type phi_1_up = make_with_random_values<phi_1_up_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: shift
  const shift_type shift = make_with_random_values<shift_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: d_phi
  const d_phi_type d_phi = make_with_random_values<d_phi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // LHS: dt_phi to be filled by manual impl
  dt_phi_type dt_phi_manual_filled(used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(
      make_not_null(&dt_phi_manual_filled), pi, d_pi, phi_two_normals,
      three_index_constraint, gamma2, phi_one_normal, phi_1_up, lapse, shift,
      d_phi);

  // LHS: dt_phi to be filled by TensorExpression impl<1>
  dt_phi_type dt_phi_te1_filled(used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&dt_phi_te1_filled), pi, d_pi, phi_two_normals,
      three_index_constraint, gamma2, phi_one_normal, phi_1_up, lapse, shift,
      d_phi);

  // CHECK dt_phi
  CHECK_ITERABLE_APPROX(dt_phi_manual_filled, dt_phi_te1_filled);

  // === Check TE impl with TempTensors ===

  size_t num_grid_points = 0;
  if constexpr (std::is_same_v<DataType, DataVector>) {
    num_grid_points = used_for_size.size();
  }

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_phi_type>, ::Tags::TempTensor<1, pi_type>,
      ::Tags::TempTensor<2, d_pi_type>,
      ::Tags::TempTensor<3, phi_two_normals_type>,
      ::Tags::TempTensor<4, three_index_constraint_type>,
      ::Tags::TempTensor<5, gamma2_type>,
      ::Tags::TempTensor<6, phi_one_normal_type>,
      ::Tags::TempTensor<7, phi_1_up_type>, ::Tags::TempTensor<8, lapse_type>,
      ::Tags::TempTensor<9, shift_type>, ::Tags::TempTensor<10, d_phi_type>>>
      vars{num_grid_points};

  // RHS: pi
  pi_type& pi_te_temp = get<::Tags::TempTensor<1, pi_type>>(vars);
  copy_tensor(pi, make_not_null(&pi_te_temp));

  // RHS: d_pi
  d_pi_type& d_pi_te_temp = get<::Tags::TempTensor<2, d_pi_type>>(vars);
  copy_tensor(d_pi, make_not_null(&d_pi_te_temp));

  // RHS: phi_two_normals
  phi_two_normals_type& phi_two_normals_te_temp =
      get<::Tags::TempTensor<3, phi_two_normals_type>>(vars);
  copy_tensor(phi_two_normals, make_not_null(&phi_two_normals_te_temp));

  // RHS: three_index_constraint
  three_index_constraint_type& three_index_constraint_te_temp =
      get<::Tags::TempTensor<4, three_index_constraint_type>>(vars);
  copy_tensor(three_index_constraint,
              make_not_null(&three_index_constraint_te_temp));

  // RHS: gamma2
  gamma2_type& gamma2_te_temp = get<::Tags::TempTensor<5, gamma2_type>>(vars);
  copy_tensor(gamma2, make_not_null(&gamma2_te_temp));

  // RHS: phi_one_normal
  phi_one_normal_type& phi_one_normal_te_temp =
      get<::Tags::TempTensor<6, phi_one_normal_type>>(vars);
  copy_tensor(phi_one_normal, make_not_null(&phi_one_normal_te_temp));

  // RHS: phi_1_up
  phi_1_up_type& phi_1_up_te_temp =
      get<::Tags::TempTensor<7, phi_1_up_type>>(vars);
  copy_tensor(phi_1_up, make_not_null(&phi_1_up_te_temp));

  // RHS: lapse
  lapse_type& lapse_te_temp = get<::Tags::TempTensor<8, lapse_type>>(vars);
  copy_tensor(lapse, make_not_null(&lapse_te_temp));

  // RHS: shift
  shift_type& shift_te_temp = get<::Tags::TempTensor<9, shift_type>>(vars);
  copy_tensor(shift, make_not_null(&shift_te_temp));

  // RHS: d_phi
  d_phi_type& d_phi_te_temp = get<::Tags::TempTensor<10, d_phi_type>>(vars);
  copy_tensor(d_phi, make_not_null(&d_phi_te_temp));

  // LHS: dt_phi impl<1>
  dt_phi_type& dt_phi_te1_temp = get<::Tags::TempTensor<0, dt_phi_type>>(vars);

  // Compute TensorExpression impl<1> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&dt_phi_te1_temp), pi_te_temp, d_pi_te_temp,
      phi_two_normals_te_temp, three_index_constraint_te_temp, gamma2_te_temp,
      phi_one_normal_te_temp, phi_1_up_te_temp, lapse_te_temp, shift_te_temp,
      d_phi_te_temp);

  // CHECK dt_phi
  CHECK_ITERABLE_APPROX(dt_phi_manual_filled, dt_phi_te1_temp);
}

template <typename DataType, typename Generator>
void test_benchmarked_impls(const DataType& used_for_size,
                            const gsl::not_null<Generator*> generator) {
  test_benchmarked_impls_core<1>(used_for_size, generator);
  test_benchmarked_impls_core<2>(used_for_size, generator);
  test_benchmarked_impls_core<3>(used_for_size, generator);
}

SPECTRE_TEST_CASE("Unit.Benchmark.GHTimeDerivative.dt_phi",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_impls(std::numeric_limits<double>::signaling_NaN(),
                         make_not_null(&generator));
  test_benchmarked_impls(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
