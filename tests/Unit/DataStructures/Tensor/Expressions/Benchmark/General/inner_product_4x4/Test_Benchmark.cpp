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
#include "Executables/Benchmark/General/inner_product_4x4/BenchmarkedImpls.hpp"
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
  using result_type = typename BenchmarkImpl::result_type;
  using R_type = typename BenchmarkImpl::R_type;
  using S_type = typename BenchmarkImpl::S_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: R
  const R_type R = make_with_random_values<R_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: S
  const S_type S = make_with_random_values<S_type>(
      generator, make_not_null(&distribution), used_for_size);

  // Compute manual result that returns LHS tensor
  const result_type result_manual_returned =
      BenchmarkImpl::manual_impl_lhs_return(R, S);

  // Compute TensorExpression result that returns LHS tensor
  const result_type result_te_returned =
      BenchmarkImpl::tensorexpression_impl_lhs_return(R, S);

  // LHS: result to be filled by manual impl
  result_type result_manual_filled(used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(make_not_null(&result_manual_filled), R,
                                     S);

  // LHS: result to be filled by TensorExpression impl<1>
  result_type result_te1_filled(used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&result_te1_filled), R, S);

  // CHECK result
  CHECK_ITERABLE_APPROX(result_manual_returned, result_te_returned);
  CHECK_ITERABLE_APPROX(result_manual_filled, result_te1_filled);

  // === Check TE impl with TempTensors ===

  size_t num_grid_points = 0;
  if constexpr (std::is_same_v<DataType, DataVector>) {
    num_grid_points = used_for_size.size();
  }

  TempBuffer<
      tmpl::list<::Tags::TempTensor<0, result_type>,
                 ::Tags::TempTensor<1, R_type>, ::Tags::TempTensor<2, S_type>>>
      vars{num_grid_points};

  // RHS: R
  R_type& R_te_temp = get<::Tags::TempTensor<1, R_type>>(vars);
  copy_tensor(R, make_not_null(&R_te_temp));

  // RHS: S
  S_type& S_te_temp = get<::Tags::TempTensor<2, S_type>>(vars);
  copy_tensor(S, make_not_null(&S_te_temp));

  // LHS: result impl<1>
  result_type& result_te1_temp = get<::Tags::TempTensor<0, result_type>>(vars);

  // Compute TensorExpression impl<1> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&result_te1_temp), R_te_temp, S_te_temp);

  // CHECK result
  CHECK_ITERABLE_APPROX(result_manual_filled, result_te1_temp);
}

template <typename DataType, typename Generator>
void test_benchmarked_impls(const DataType& used_for_size,
                            const gsl::not_null<Generator*> generator) {
  test_benchmarked_impls_core<1>(used_for_size, generator);
  test_benchmarked_impls_core<2>(used_for_size, generator);
  test_benchmarked_impls_core<3>(used_for_size, generator);
}

SPECTRE_TEST_CASE("Unit.Benchmark.General.inner_product_4x4",
                  "[Unit][DataStructures]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_impls(std::numeric_limits<double>::signaling_NaN(),
                         make_not_null(&generator));
  test_benchmarked_impls(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
