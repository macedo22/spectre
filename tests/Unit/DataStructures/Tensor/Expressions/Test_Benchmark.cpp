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
  using normal_dot_gauge_constraint_type =
      typename BenchmarkImpl::normal_dot_gauge_constraint_type;
  using normal_spacetime_vector_type =
      typename BenchmarkImpl::normal_spacetime_vector_type;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: normal_spacetime_vector
  const normal_spacetime_vector_type normal_spacetime_vector =
      make_with_random_values<normal_spacetime_vector_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: gauge_constraint
  const gauge_constraint_type gauge_constraint =
      make_with_random_values<gauge_constraint_type>(
          generator, make_not_null(&distribution), used_for_size);

  // Compute manual result that returns LHS tensor
  const normal_dot_gauge_constraint_type
      normal_dot_gauge_constraint_manual_returned =
          BenchmarkImpl::manual_impl_lhs_return(normal_spacetime_vector,
                                                gauge_constraint);

  // Compute TensorExpression result that returns LHS tensor
  const normal_dot_gauge_constraint_type
      normal_dot_gauge_constraint_te_returned =
          BenchmarkImpl::tensorexpression_impl_lhs_return(
              normal_spacetime_vector, gauge_constraint);

  // LHS: normal_dot_gauge_constraint to be filled by manual impl
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint_manual_filled(
      used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(
      make_not_null(&normal_dot_gauge_constraint_manual_filled),
      normal_spacetime_vector, gauge_constraint);

  // LHS: normal_dot_gauge_constraint to be filled by TensorExpression impl<1>
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint_te1_filled(
      used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&normal_dot_gauge_constraint_te1_filled),
      normal_spacetime_vector, gauge_constraint);

  // CHECK normal_dot_gauge_constraint
  CHECK_ITERABLE_APPROX(normal_dot_gauge_constraint_manual_returned.get(),
                        normal_dot_gauge_constraint_te_returned.get());
  CHECK_ITERABLE_APPROX(normal_dot_gauge_constraint_manual_filled.get(),
                        normal_dot_gauge_constraint_te1_filled.get());

  // For DataVectors, check TE impl with TempTensors
  if constexpr (not std::is_same_v<DataType, double>) {
    TempBuffer<
        tmpl::list<::Tags::TempTensor<0, normal_dot_gauge_constraint_type>,
                   ::Tags::TempTensor<1, normal_spacetime_vector_type>,
                   ::Tags::TempTensor<2, gauge_constraint_type>>>
        vars{used_for_size.size()};

    // RHS: normal_spacetime_vector
    normal_spacetime_vector_type& normal_spacetime_vector_te_temp =
        get<::Tags::TempTensor<1, normal_spacetime_vector_type>>(vars);
    copy_tensor(normal_spacetime_vector,
                make_not_null(&normal_spacetime_vector_te_temp));

    // RHS: gauge_constraint
    gauge_constraint_type& gauge_constraint_te_temp =
        get<::Tags::TempTensor<2, gauge_constraint_type>>(vars);
    copy_tensor(gauge_constraint, make_not_null(&gauge_constraint_te_temp));

    // LHS: normal_dot_gauge_constraint impl<1>
    normal_dot_gauge_constraint_type& normal_dot_gauge_constraint_te1_temp =
        get<::Tags::TempTensor<0, normal_dot_gauge_constraint_type>>(vars);

    // Compute TensorExpression impl<1> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
        make_not_null(&normal_dot_gauge_constraint_te1_temp),
        normal_spacetime_vector, gauge_constraint_te_temp);

    // CHECK normal_dot_gauge_constraint
    CHECK_ITERABLE_APPROX(normal_dot_gauge_constraint_manual_filled.get(),
                          normal_dot_gauge_constraint_te1_temp.get());
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
