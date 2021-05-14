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
#include "Executables/Benchmark/BenchmarkedImpls.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
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
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: trace_christoffel
  const trace_christoffel_type trace_christoffel =
      make_with_random_values<trace_christoffel_type>(
          generator, make_not_null(&distribution), used_for_size);

  // Compute manual result that returns LHS tensor
  const gauge_constraint_type gauge_constraint_manual_returned =
      BenchmarkImpl::manual_impl_lhs_return(gauge_function, trace_christoffel);

  // Compute TensorExpression result that returns LHS tensor
  const gauge_constraint_type gauge_constraint_te_returned =
      BenchmarkImpl::tensorexpression_impl_lhs_return(gauge_function,
                                                      trace_christoffel);

  // LHS: gauge_constraint to be filled by manual impl
  gauge_constraint_type gauge_constraint_manual_filled(used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(
      make_not_null(&gauge_constraint_manual_filled), gauge_function,
      trace_christoffel);

  // LHS: gauge_constraint to be filled by TensorExpression impl<1>
  gauge_constraint_type gauge_constraint_te1_filled(used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&gauge_constraint_te1_filled), gauge_function,
      trace_christoffel);

  // CHECK gauge_constraint
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(gauge_constraint_manual_returned.get(a),
                          gauge_constraint_te_returned.get(a));
    CHECK_ITERABLE_APPROX(gauge_constraint_manual_filled.get(a),
                          gauge_constraint_te1_filled.get(a));
  }

  // For DataVectors, check TE impl with TempTensors
  if constexpr (not std::is_same_v<DataType, double>) {
    TempBuffer<tmpl::list<::Tags::TempTensor<0, gauge_constraint_type>,
                          ::Tags::TempTensor<1, gauge_function_type>,
                          ::Tags::TempTensor<2, trace_christoffel_type>>>
        vars{used_for_size.size()};

    // RHS: gauge_function
    gauge_function_type& gauge_function_te_temp =
        get<::Tags::TempTensor<1, gauge_function_type>>(vars);
    copy_tensor(gauge_function, make_not_null(&gauge_function_te_temp));

    // RHS: trace_christoffel
    trace_christoffel_type& trace_christoffel_te_temp =
        get<::Tags::TempTensor<2, trace_christoffel_type>>(vars);
    copy_tensor(trace_christoffel, make_not_null(&trace_christoffel_te_temp));

    // LHS: gauge_constraint impl<1>
    gauge_constraint_type& gauge_constraint_te1_temp =
        get<::Tags::TempTensor<0, gauge_constraint_type>>(vars);

    // Compute TensorExpression impl<1> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
        make_not_null(&gauge_constraint_te1_temp), gauge_function,
        trace_christoffel_te_temp);

    // CHECK gauge_constraint
    for (size_t a = 0; a < Dim + 1; a++) {
      CHECK_ITERABLE_APPROX(gauge_constraint_manual_filled.get(a),
                            gauge_constraint_te1_temp.get(a));
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

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Benchmark",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_impls(std::numeric_limits<double>::signaling_NaN(),
                         make_not_null(&generator));
  test_benchmarked_impls(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
