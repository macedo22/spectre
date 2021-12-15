// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Executables/Benchmark/DataVector/InnerProductImpls.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

// Make sure TE impl matches manual impl
template <typename Generator, typename DataType>
void test_benchmarked_impls(const gsl::not_null<Generator*> generator,
                            const DataType& used_for_size) {
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

  // LHS: result inner product that uses loop
  result_type result_loop(used_for_size);

  // Compute result inner product that uses loop
  BenchmarkImpl::loop_impl(make_not_null(&result_loop), R, S);

  // LHS: result inner product that uses one line
  result_type result_oneline(used_for_size);

  // Compute result inner product that uses one line
  BenchmarkImpl::oneline_impl(make_not_null(&result_oneline), R, S);

  // LHS: result inner product that uses recursion
  result_type result_recursive(used_for_size);

  // Compute result inner product that uses recursion
  BenchmarkImpl::recursive_impl(make_not_null(&result_recursive), R, S);

  // LHS: result inner product that uses recursion with +=
  result_type result_recursive_plus_equals(used_for_size);

  // Compute result inner product that uses recursion with +=
  BenchmarkImpl::recursive_plus_equals_impl(
      make_not_null(&result_recursive_plus_equals), R, S);

  // CHECK all inner products match
  CHECK_ITERABLE_APPROX(result_loop, result_oneline);
  CHECK_ITERABLE_APPROX(result_loop, result_recursive);
  CHECK_ITERABLE_APPROX(result_loop, result_recursive_plus_equals);
}

SPECTRE_TEST_CASE("Unit.Benchmark.DataVector.inner_product_spacetime_4x4",
                  "[Unit][DataStructures]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_impls(
      make_not_null(&generator),
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
