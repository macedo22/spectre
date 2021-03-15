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

  using phi_1_up_type = BenchmarkImpl::phi_1_up_type;
  using inverse_spatial_metric_type =
      BenchmarkImpl::inverse_spatial_metric_type;
  using phi_type = BenchmarkImpl::phi_type;

  // inverse_spatial_metric
  inverse_spatial_metric_type inverse_spatial_metric(num_grid_points);
  assign_unique_values_to_tensor(make_not_null(&inverse_spatial_metric));

  // phi
  phi_type phi(num_grid_points);
  assign_unique_values_to_tensor(make_not_null(&phi));

  // LHS: phi_1_up for manual implementation
  phi_1_up_type phi_1_up_manual(num_grid_points);

  BenchmarkImpl::manual_impl(make_not_null(&phi_1_up_manual),
                             make_not_null(&inverse_spatial_metric), phi);

  // LHS: phi_1_up for TE implementation with LHS returned
  const phi_1_up_type phi_1_up_te_returned =
      BenchmarkImpl::tensorexpression_impl_return(
          make_not_null(&inverse_spatial_metric), phi);

  // LHS: phi_1_up for TE implementation with LHS pass as arg
  phi_1_up_type phi_1_up_te_filled(num_grid_points);
  BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
      make_not_null(&phi_1_up_te_filled),
      make_not_null(&inverse_spatial_metric), phi);

  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_1_up_manual.get(i, a, b),
                              phi_1_up_te_returned.get(i, a, b));
        CHECK_ITERABLE_APPROX(phi_1_up_manual.get(i, a, b),
                              phi_1_up_te_filled.get(i, a, b));
      }
    }
  }

  if constexpr (not std::is_same_v<DataType, double>) {
    TempBuffer<tmpl::list<::Tags::TempTensor<0, phi_1_up_type>,
                          ::Tags::TempTensor<1, inverse_spatial_metric_type>,
                          ::Tags::TempTensor<2, phi_type>>>
        vars{num_grid_points};

    // inverse_spatial_metric
    inverse_spatial_metric_type& inverse_spatial_metric_te_temp =
        get<::Tags::TempTensor<1, inverse_spatial_metric_type>>(vars);
    assign_unique_values_to_tensor(
        make_not_null(&inverse_spatial_metric_te_temp));

    // phi
    phi_type& phi_te_temp = get<::Tags::TempTensor<2, phi_type>>(vars);
    assign_unique_values_to_tensor(make_not_null(&phi_te_temp));

    // LHS: phi_1_up
    phi_1_up_type& phi_1_up_te_temp =
        get<::Tags::TempTensor<0, phi_1_up_type>>(vars);
    BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
        make_not_null(&phi_1_up_te_temp),
        make_not_null(&inverse_spatial_metric_te_temp), phi_te_temp);

    for (size_t i = 0; i < Dim; i++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = 0; b < Dim + 1; b++) {
          CHECK_ITERABLE_APPROX(phi_1_up_manual.get(i, a, b),
                                phi_1_up_te_temp.get(i, a, b));
        }
      }
    }
  }
}
