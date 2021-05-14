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
#include "Executables/Benchmark/GHTimeDerivative/christoffel_second_kind/BenchmarkedImpls.hpp"
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
  using christoffel_second_kind_type =
      typename BenchmarkImpl::christoffel_second_kind_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: christoffel_first_kind
  const christoffel_first_kind_type christoffel_first_kind =
      make_with_random_values<christoffel_first_kind_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: inverse_spacetime_metric
  const inverse_spacetime_metric_type inverse_spacetime_metric =
      make_with_random_values<inverse_spacetime_metric_type>(
          generator, make_not_null(&distribution), used_for_size);

  // Compute manual result that returns LHS tensor
  const christoffel_second_kind_type christoffel_second_kind_manual_returned =
      BenchmarkImpl::manual_impl_lhs_return(christoffel_first_kind,
                                            inverse_spacetime_metric);

  // Compute TensorExpression result that returns LHS tensor
  const christoffel_second_kind_type christoffel_second_kind_te_returned =
      BenchmarkImpl::tensorexpression_impl_lhs_return(christoffel_first_kind,
                                                      inverse_spacetime_metric);

  // LHS: christoffel_second_kind to be filled by manual impl
  christoffel_second_kind_type christoffel_second_kind_manual_filled(
      used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(
      make_not_null(&christoffel_second_kind_manual_filled),
      christoffel_first_kind, inverse_spacetime_metric);

  // LHS: christoffel_second_kind to be filled by TensorExpression impl<1>
  christoffel_second_kind_type christoffel_second_kind_te1_filled(
      used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&christoffel_second_kind_te1_filled),
      christoffel_first_kind, inverse_spacetime_metric);

  // LHS: christoffel_second_kind to be filled by TensorExpression impl<2>
  christoffel_second_kind_type christoffel_second_kind_te2_filled(
      used_for_size);

  // Compute TensorExpression impl<2> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<2>(
      make_not_null(&christoffel_second_kind_te2_filled),
      christoffel_first_kind, inverse_spacetime_metric);

  // LHS: christoffel_second_kind to be filled by TensorExpression impl<3>
  christoffel_second_kind_type christoffel_second_kind_te3_filled(
      used_for_size);

  // Compute TensorExpression impl<3> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<3>(
      make_not_null(&christoffel_second_kind_te3_filled),
      christoffel_first_kind, inverse_spacetime_metric);

  // LHS: christoffel_second_kind to be filled by TensorExpression impl<4>
  christoffel_second_kind_type christoffel_second_kind_te4_filled(
      used_for_size);

  // Compute TensorExpression impl<4> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<4>(
      make_not_null(&christoffel_second_kind_te4_filled),
      christoffel_first_kind, inverse_spacetime_metric);

  // LHS: christoffel_second_kind to be filled by TensorExpression impl<5>
  christoffel_second_kind_type christoffel_second_kind_te5_filled(
      used_for_size);

  // Compute TensorExpression impl<4> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<5>(
      make_not_null(&christoffel_second_kind_te5_filled),
      christoffel_first_kind, inverse_spacetime_metric);

  // CHECK christoffel_second_kind (iaa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_returned.get(a, b, c),
            christoffel_second_kind_te_returned.get(a, b, c));
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te1_filled.get(a, b, c));
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te2_filled.get(a, b, c));
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te3_filled.get(a, b, c));
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te4_filled.get(a, b, c));
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te5_filled.get(a, b, c));
      }
    }
  }

  // === Check TE impl with TempTensors ===

  size_t num_grid_points = 0;
  if constexpr (std::is_same_v<DataType, DataVector>) {
    num_grid_points = used_for_size.size();
  }

  TempBuffer<tmpl::list<::Tags::TempTensor<0, christoffel_second_kind_type>,
                        ::Tags::TempTensor<1, christoffel_second_kind_type>,
                        ::Tags::TempTensor<2, christoffel_second_kind_type>,
                        ::Tags::TempTensor<3, christoffel_second_kind_type>,
                        ::Tags::TempTensor<4, christoffel_second_kind_type>,
                        ::Tags::TempTensor<5, christoffel_first_kind_type>,
                        ::Tags::TempTensor<6, inverse_spacetime_metric_type>>>
      vars{num_grid_points};

  // RHS: christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind_te_temp =
      get<::Tags::TempTensor<5, christoffel_first_kind_type>>(vars);
  copy_tensor(christoffel_first_kind,
              make_not_null(&christoffel_first_kind_te_temp));

  // RHS: inverse_spacetime_metric
  inverse_spacetime_metric_type& inverse_spacetime_metric_te_temp =
      get<::Tags::TempTensor<6, inverse_spacetime_metric_type>>(vars);
  copy_tensor(inverse_spacetime_metric,
              make_not_null(&inverse_spacetime_metric_te_temp));

  // LHS: christoffel_second_kind impl<1>
  christoffel_second_kind_type& christoffel_second_kind_te1_temp =
      get<::Tags::TempTensor<0, christoffel_second_kind_type>>(vars);

  // Compute TensorExpression impl<1> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&christoffel_second_kind_te1_temp),
      christoffel_first_kind_te_temp, inverse_spacetime_metric_te_temp);

  // LHS: christoffel_second_kind impl<2>
  christoffel_second_kind_type& christoffel_second_kind_te2_temp =
      get<::Tags::TempTensor<1, christoffel_second_kind_type>>(vars);

  // Compute TensorExpression impl<2> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<2>(
      make_not_null(&christoffel_second_kind_te2_temp),
      christoffel_first_kind_te_temp, inverse_spacetime_metric_te_temp);

  // LHS: christoffel_second_kind impl<3>
  christoffel_second_kind_type& christoffel_second_kind_te3_temp =
      get<::Tags::TempTensor<2, christoffel_second_kind_type>>(vars);

  // Compute TensorExpression impl<3> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<3>(
      make_not_null(&christoffel_second_kind_te3_temp),
      christoffel_first_kind_te_temp, inverse_spacetime_metric_te_temp);

  // LHS: christoffel_second_kind impl<4>
  christoffel_second_kind_type& christoffel_second_kind_te4_temp =
      get<::Tags::TempTensor<3, christoffel_second_kind_type>>(vars);

  // Compute TensorExpression impl<4> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<4>(
      make_not_null(&christoffel_second_kind_te4_temp),
      christoffel_first_kind_te_temp, inverse_spacetime_metric_te_temp);

  // LHS: christoffel_second_kind impl<5>
  christoffel_second_kind_type& christoffel_second_kind_te5_temp =
      get<::Tags::TempTensor<4, christoffel_second_kind_type>>(vars);

  // Compute TensorExpression impl<5> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<5>(
      make_not_null(&christoffel_second_kind_te5_temp),
      christoffel_first_kind_te_temp, inverse_spacetime_metric_te_temp);

  // CHECK christoffel_second_kind
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te1_temp.get(a, b, c));
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te2_temp.get(a, b, c));
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te3_temp.get(a, b, c));
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te4_temp.get(a, b, c));
        CHECK_ITERABLE_APPROX(
            christoffel_second_kind_manual_filled.get(a, b, c),
            christoffel_second_kind_te5_temp.get(a, b, c));
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

SPECTRE_TEST_CASE("Unit.Benchmark.GHTimeDerivative.christoffel_second_kind",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_impls(std::numeric_limits<double>::signaling_NaN(),
                         make_not_null(&generator));
  test_benchmarked_impls(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
