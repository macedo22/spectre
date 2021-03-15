// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Executables/Benchmark/BenchmarkedImpls.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Benchmark",
                  "[DataStructures][Unit]") {
  // Make sure TE impl matches manual impl

  constexpr size_t Dim = BenchmarkImpl::Dim;
  using DataType = BenchmarkImpl::DataType;
  constexpr size_t num_grid_points = 5;

  using christoffel_second_kind_type =
      BenchmarkImpl::christoffel_second_kind_type;
  using christoffel_first_kind_type =
      BenchmarkImpl::christoffel_first_kind_type;
  using inverse_spacetime_metric_type =
      BenchmarkImpl::inverse_spacetime_metric_type;

  // christoffel_first_kind
  christoffel_first_kind_type christoffel_first_kind(num_grid_points);
  assign_unique_values_to_tensor(make_not_null(&christoffel_first_kind));

  // inverse_spacetime_metric
  inverse_spacetime_metric_type inverse_spacetime_metric(num_grid_points);
  assign_unique_values_to_tensor(make_not_null(&inverse_spacetime_metric));

  // LHS: christoffel_second_kind for manual implementation
  christoffel_second_kind_type christoffel_second_kind_manual(num_grid_points);

  BenchmarkImpl::manual_impl(make_not_null(&christoffel_second_kind_manual),
                             christoffel_first_kind, inverse_spacetime_metric);

  // LHS: christoffel_second_kind for TE implementation with LHS returned
  const christoffel_second_kind_type christoffel_second_kind_te_returned =
      BenchmarkImpl::tensorexpression_impl_return(christoffel_first_kind,
                                                  inverse_spacetime_metric);

  // LHS: christoffel_second_kind for TE implementation with LHS pass as arg
  christoffel_second_kind_type christoffel_second_kind_te_filled(
      num_grid_points);
  BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
      make_not_null(&christoffel_second_kind_te_filled), christoffel_first_kind,
      inverse_spacetime_metric);

  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(christoffel_second_kind_manual.get(a, b, c),
                              christoffel_second_kind_te_returned.get(a, b, c));
        CHECK_ITERABLE_APPROX(christoffel_second_kind_manual.get(a, b, c),
                              christoffel_second_kind_te_filled.get(a, b, c));
      }
    }
  }

  if constexpr (not std::is_same_v<DataType, double>) {
    TempBuffer<tmpl::list<::Tags::TempTensor<0, christoffel_second_kind_type>,
                          ::Tags::TempTensor<1, christoffel_first_kind_type>,
                          ::Tags::TempTensor<2, inverse_spacetime_metric_type>>>
        vars{num_grid_points};

    // christoffel_first_kind
    christoffel_first_kind_type& christoffel_first_kind_te_temp =
        get<::Tags::TempTensor<1, christoffel_first_kind_type>>(vars);
    assign_unique_values_to_tensor(
        make_not_null(&christoffel_first_kind_te_temp));

    // inverse_spacetime_metric
    inverse_spacetime_metric_type& inverse_spacetime_metric_te_temp =
        get<::Tags::TempTensor<2, inverse_spacetime_metric_type>>(vars);
    assign_unique_values_to_tensor(
        make_not_null(&inverse_spacetime_metric_te_temp));

    // LHS: christoffel_second_kind
    christoffel_second_kind_type& christoffel_second_kind_te_temp =
        get<::Tags::TempTensor<0, christoffel_second_kind_type>>(vars);
    BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
        make_not_null(&christoffel_second_kind_te_temp),
        christoffel_first_kind_te_temp, inverse_spacetime_metric_te_temp);

    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        for (size_t c = 0; c < Dim + 1; c++) {
          CHECK_ITERABLE_APPROX(christoffel_second_kind_manual.get(a, b, c),
                                christoffel_second_kind_te_temp.get(a, b, c));
        }
      }
    }
  }
}
