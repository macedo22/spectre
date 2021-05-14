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
#include "Executables/Benchmark/GHTimeDerivative/pi_one_normal/BenchmarkedImpls.hpp"
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
  using pi_one_normal_type = typename BenchmarkImpl::pi_one_normal_type;
  using normal_spacetime_vector_type =
      typename BenchmarkImpl::normal_spacetime_vector_type;
  using pi_type = typename BenchmarkImpl::pi_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: normal_spacetime_vector
  const normal_spacetime_vector_type normal_spacetime_vector =
      make_with_random_values<normal_spacetime_vector_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // Compute manual result that returns LHS tensor
  const pi_one_normal_type pi_one_normal_manual_returned =
      BenchmarkImpl::manual_impl_lhs_return(normal_spacetime_vector, pi);

  // Compute TensorExpression result that returns LHS tensor
  const pi_one_normal_type pi_one_normal_te_returned =
      BenchmarkImpl::tensorexpression_impl_lhs_return(normal_spacetime_vector,
                                                      pi);

  // LHS: pi_one_normal to be filled by manual impl
  pi_one_normal_type pi_one_normal_manual_filled(used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(
      make_not_null(&pi_one_normal_manual_filled), normal_spacetime_vector, pi);

  // LHS: pi_one_normal to be filled by TensorExpression impl<1>
  pi_one_normal_type pi_one_normal_te1_filled(used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&pi_one_normal_te1_filled), normal_spacetime_vector, pi);

  // LHS: pi_one_normal to be filled by TensorExpression impl<2>
  pi_one_normal_type pi_one_normal_te2_filled(used_for_size);

  // Compute TensorExpression impl<2> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<2>(
      make_not_null(&pi_one_normal_te2_filled), normal_spacetime_vector, pi);

  // LHS: pi_one_normal to be filled by TensorExpression impl<3>
  pi_one_normal_type pi_one_normal_te3_filled(used_for_size);

  // Compute TensorExpression impl<3> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<3>(
      make_not_null(&pi_one_normal_te3_filled), normal_spacetime_vector, pi);

  // LHS: pi_one_normal to be filled by TensorExpression impl<4>
  pi_one_normal_type pi_one_normal_te4_filled(used_for_size);

  // Compute TensorExpression impl<4> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<4>(
      make_not_null(&pi_one_normal_te4_filled), normal_spacetime_vector, pi);

  // CHECK pi_one_normal (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_returned.get(a),
                          pi_one_normal_te_returned.get(a));
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_filled.get(a),
                          pi_one_normal_te1_filled.get(a));
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_filled.get(a),
                          pi_one_normal_te2_filled.get(a));
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_filled.get(a),
                          pi_one_normal_te3_filled.get(a));
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_filled.get(a),
                          pi_one_normal_te4_filled.get(a));
  }

  // === Check TE impl with TempTensors ===

  size_t num_grid_points = 0;
  if constexpr (std::is_same_v<DataType, DataVector>) {
    num_grid_points = used_for_size.size();
  }

  TempBuffer<tmpl::list<::Tags::TempTensor<0, pi_one_normal_type>,
                        ::Tags::TempTensor<1, pi_one_normal_type>,
                        ::Tags::TempTensor<2, pi_one_normal_type>,
                        ::Tags::TempTensor<3, pi_one_normal_type>,
                        ::Tags::TempTensor<4, normal_spacetime_vector_type>,
                        ::Tags::TempTensor<5, pi_type>>>
      vars{num_grid_points};

  // RHS: normal_spacetime_vector
  normal_spacetime_vector_type& normal_spacetime_vector_te_temp =
      get<::Tags::TempTensor<4, normal_spacetime_vector_type>>(vars);
  copy_tensor(normal_spacetime_vector,
              make_not_null(&normal_spacetime_vector_te_temp));

  // RHS: pi
  pi_type& pi_te_temp = get<::Tags::TempTensor<5, pi_type>>(vars);
  copy_tensor(pi, make_not_null(&pi_te_temp));

  // LHS: pi_one_normal impl<1>
  pi_one_normal_type& pi_one_normal_te1_temp =
      get<::Tags::TempTensor<0, pi_one_normal_type>>(vars);

  // Compute TensorExpression impl<1> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&pi_one_normal_te1_temp), normal_spacetime_vector_te_temp,
      pi_te_temp);

  // LHS: pi_one_normal impl<2>
  pi_one_normal_type& pi_one_normal_te2_temp =
      get<::Tags::TempTensor<1, pi_one_normal_type>>(vars);

  // Compute TensorExpression impl<2> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<2>(
      make_not_null(&pi_one_normal_te2_temp), normal_spacetime_vector_te_temp,
      pi_te_temp);

  // LHS: pi_one_normal impl<3>
  pi_one_normal_type& pi_one_normal_te3_temp =
      get<::Tags::TempTensor<2, pi_one_normal_type>>(vars);

  // Compute TensorExpression impl<3> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<3>(
      make_not_null(&pi_one_normal_te3_temp), normal_spacetime_vector_te_temp,
      pi_te_temp);

  // LHS: pi_one_normal impl<4>
  pi_one_normal_type& pi_one_normal_te4_temp =
      get<::Tags::TempTensor<3, pi_one_normal_type>>(vars);

  // Compute TensorExpression impl<4> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<4>(
      make_not_null(&pi_one_normal_te4_temp), normal_spacetime_vector_te_temp,
      pi_te_temp);

  // CHECK pi_one_normal (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_filled.get(a),
                          pi_one_normal_te1_temp.get(a));
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_filled.get(a),
                          pi_one_normal_te2_temp.get(a));
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_filled.get(a),
                          pi_one_normal_te3_temp.get(a));
    CHECK_ITERABLE_APPROX(pi_one_normal_manual_filled.get(a),
                          pi_one_normal_te4_temp.get(a));
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

SPECTRE_TEST_CASE("Unit.Benchmark.GHTimeDerivative.pi_one_normal",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_impls(std::numeric_limits<double>::signaling_NaN(),
                         make_not_null(&generator));
  test_benchmarked_impls(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
