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
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using d_spacetime_metric_type =
      typename BenchmarkImpl::d_spacetime_metric_type;
  using phi_type = typename BenchmarkImpl::phi_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: d_spacetime_metric
  const d_spacetime_metric_type d_spacetime_metric =
      make_with_random_values<d_spacetime_metric_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: phi
  const phi_type phi = make_with_random_values<phi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // Compute manual result that returns LHS tensor
  const three_index_constraint_type three_index_constraint_manual_returned =
      BenchmarkImpl::manual_impl_lhs_return(d_spacetime_metric, phi);

  // Compute TensorExpression result that returns LHS tensor
  const three_index_constraint_type three_index_constraint_te_returned =
      BenchmarkImpl::tensorexpression_impl_lhs_return(d_spacetime_metric, phi);

  // LHS: three_index_constraint to be filled by manual impl
  three_index_constraint_type three_index_constraint_manual_filled(
      used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(
      make_not_null(&three_index_constraint_manual_filled), d_spacetime_metric,
      phi);

  // LHS: three_index_constraint to be filled by TensorExpression impl<1>
  three_index_constraint_type three_index_constraint_te1_filled(used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&three_index_constraint_te1_filled), d_spacetime_metric,
      phi);

  // LHS: three_index_constraint to be filled by TensorExpression impl<2>
  three_index_constraint_type three_index_constraint_te2_filled(used_for_size);

  // Compute TensorExpression impl<2> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<2>(
      make_not_null(&three_index_constraint_te2_filled), d_spacetime_metric,
      phi);

  // LHS: three_index_constraint to be filled by TensorExpression impl<3>
  three_index_constraint_type three_index_constraint_te3_filled(used_for_size);

  // Compute TensorExpression impl<3> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<3>(
      make_not_null(&three_index_constraint_te3_filled), d_spacetime_metric,
      phi);

  // LHS: three_index_constraint to be filled by TensorExpression impl<4>
  three_index_constraint_type three_index_constraint_te4_filled(used_for_size);

  // Compute TensorExpression impl<4> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<4>(
      make_not_null(&three_index_constraint_te4_filled), d_spacetime_metric,
      phi);

  // CHECK three_index_constraint (iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(
            three_index_constraint_manual_returned.get(i, a, b),
            three_index_constraint_te_returned.get(i, a, b));
        CHECK_ITERABLE_APPROX(three_index_constraint_manual_filled.get(i, a, b),
                              three_index_constraint_te1_filled.get(i, a, b));
        CHECK_ITERABLE_APPROX(three_index_constraint_manual_filled.get(i, a, b),
                              three_index_constraint_te2_filled.get(i, a, b));
        CHECK_ITERABLE_APPROX(three_index_constraint_manual_filled.get(i, a, b),
                              three_index_constraint_te3_filled.get(i, a, b));
        CHECK_ITERABLE_APPROX(three_index_constraint_manual_filled.get(i, a, b),
                              three_index_constraint_te4_filled.get(i, a, b));
      }
    }
  }

  // For DataVectors, check TE impl with TempTensors
  if constexpr (not std::is_same_v<DataType, double>) {
    TempBuffer<tmpl::list<::Tags::TempTensor<0, three_index_constraint_type>,
                          ::Tags::TempTensor<1, three_index_constraint_type>,
                          ::Tags::TempTensor<2, three_index_constraint_type>,
                          ::Tags::TempTensor<3, three_index_constraint_type>,
                          ::Tags::TempTensor<4, d_spacetime_metric_type>,
                          ::Tags::TempTensor<5, phi_type>>>
        vars{used_for_size.size()};

    // RHS: d_spacetime_metric
    d_spacetime_metric_type& d_spacetime_metric_te_temp =
        get<::Tags::TempTensor<4, d_spacetime_metric_type>>(vars);
    copy_tensor(d_spacetime_metric, make_not_null(&d_spacetime_metric_te_temp));

    // RHS: phi
    phi_type& phi_te_temp = get<::Tags::TempTensor<5, phi_type>>(vars);
    copy_tensor(phi, make_not_null(&phi_te_temp));

    // LHS: three_index_constraint impl<1>
    three_index_constraint_type& three_index_constraint_te1_temp =
        get<::Tags::TempTensor<0, three_index_constraint_type>>(vars);

    // Compute TensorExpression impl<1> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
        make_not_null(&three_index_constraint_te1_temp), d_spacetime_metric,
        phi_te_temp);

    // LHS: three_index_constraint impl<2>
    three_index_constraint_type& three_index_constraint_te2_temp =
        get<::Tags::TempTensor<1, three_index_constraint_type>>(vars);

    // Compute TensorExpression impl<2> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<2>(
        make_not_null(&three_index_constraint_te2_temp), d_spacetime_metric,
        phi_te_temp);

    // LHS: three_index_constraint impl<3>
    three_index_constraint_type& three_index_constraint_te3_temp =
        get<::Tags::TempTensor<2, three_index_constraint_type>>(vars);

    // Compute TensorExpression impl<3> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<3>(
        make_not_null(&three_index_constraint_te3_temp), d_spacetime_metric,
        phi_te_temp);

    // LHS: three_index_constraint impl<4>
    three_index_constraint_type& three_index_constraint_te4_temp =
        get<::Tags::TempTensor<3, three_index_constraint_type>>(vars);

    // Compute TensorExpression impl<4> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<4>(
        make_not_null(&three_index_constraint_te4_temp), d_spacetime_metric,
        phi_te_temp);

    // CHECK three_index_constraint (iaa)
    for (size_t i = 0; i < Dim; i++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = 0; b < Dim + 1; b++) {
          CHECK_ITERABLE_APPROX(
              three_index_constraint_manual_filled.get(i, a, b),
              three_index_constraint_te1_temp.get(i, a, b));
          CHECK_ITERABLE_APPROX(
              three_index_constraint_manual_filled.get(i, a, b),
              three_index_constraint_te2_temp.get(i, a, b));
          CHECK_ITERABLE_APPROX(
              three_index_constraint_manual_filled.get(i, a, b),
              three_index_constraint_te3_temp.get(i, a, b));
          CHECK_ITERABLE_APPROX(
              three_index_constraint_manual_filled.get(i, a, b),
              three_index_constraint_te4_temp.get(i, a, b));
        }
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

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Benchmark",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_impls(std::numeric_limits<double>::signaling_NaN(),
                         make_not_null(&generator));
  test_benchmarked_impls(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
