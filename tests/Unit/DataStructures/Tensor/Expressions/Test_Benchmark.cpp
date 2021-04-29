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
void test_benchmarked_implementations_core(
    const DataType& used_for_size,
    const gsl::not_null<Generator*> generator) noexcept {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: da_spacetime_metric
  const da_spacetime_metric_type da_spacetime_metric =
      make_with_random_values<da_spacetime_metric_type>(
          generator, make_not_null(&distribution), used_for_size);

  // Note: TE can't determine correct LHS symmetry, commented out

  // Compute manual result that returns LHS tensor
  // const christoffel_first_kind_type christoffel_first_kind_manual_returned =
  //     BenchmarkImpl::manual_impl_lhs_return(da_spacetime_metric);

  // Compute TensorExpression result that returns LHS tensor
  // const christoffel_first_kind_type christoffel_first_kind_te_returned =
  //     BenchmarkImpl::tensorexpression_impl_lhs_return(da_spacetime_metric);

  // LHS: christoffel_first_kind to be filled by manual implementation
  christoffel_first_kind_type christoffel_first_kind_manual_filled(
      used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(
      make_not_null(&christoffel_first_kind_manual_filled),
      da_spacetime_metric);

  // LHS: christoffel_first_kind to be filled by TensorExpression implementation
  christoffel_first_kind_type christoffel_first_kind_te_filled(used_for_size);

  // Compute TensorExpression result with LHS tensor as argument
  BenchmarkImpl::tensorexpression_impl_lhs_arg(
      make_not_null(&christoffel_first_kind_te_filled), da_spacetime_metric);

  // CHECK christoffel_first_kind (abb)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        // CHECK_ITERABLE_APPROX(
        //     christoffel_first_kind_manual_returned.get(a, b, c),
        //     christoffel_first_kind_te_returned.get(a, b, c));
        CHECK_ITERABLE_APPROX(christoffel_first_kind_manual_filled.get(a, b, c),
                              christoffel_first_kind_te_filled.get(a, b, c));
      }
    }
  }

  // For DataVectors, check TE implementation with TempTensors
  if constexpr (not std::is_same_v<DataType, double>) {
    TempBuffer<tmpl::list<::Tags::TempTensor<0, christoffel_first_kind_type>,
                          ::Tags::TempTensor<1, da_spacetime_metric_type>>>
        vars{used_for_size.size()};

    // RHS: da_spacetime_metric
    da_spacetime_metric_type& da_spacetime_metric_te_temp =
        get<::Tags::TempTensor<1, da_spacetime_metric_type>>(vars);
    copy_tensor(da_spacetime_metric,
                make_not_null(&da_spacetime_metric_te_temp));

    // LHS: christoffel_first_kind
    christoffel_first_kind_type& christoffel_first_kind_te_temp =
        get<::Tags::TempTensor<0, christoffel_first_kind_type>>(vars);

    // Compute TensorExpression result
    BenchmarkImpl::tensorexpression_impl_lhs_arg(
        make_not_null(&christoffel_first_kind_te_temp),
        da_spacetime_metric_te_temp);

    // CHECK christoffel_first_kind (abb)
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        for (size_t c = 0; c < Dim + 1; c++) {
          CHECK_ITERABLE_APPROX(
              christoffel_first_kind_manual_filled.get(a, b, c),
              christoffel_first_kind_te_temp.get(a, b, c));
        }
      }
    }
  }
}

template <typename DataType, typename Generator>
void test_benchmarked_implementations(
    const DataType& used_for_size,
    const gsl::not_null<Generator*> generator) noexcept {
  test_benchmarked_implementations_core<1>(used_for_size, generator);
  test_benchmarked_implementations_core<2>(used_for_size, generator);
  test_benchmarked_implementations_core<3>(used_for_size, generator);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Benchmark",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_implementations(std::numeric_limits<double>::signaling_NaN(),
                                   make_not_null(&generator));
  test_benchmarked_implementations(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
