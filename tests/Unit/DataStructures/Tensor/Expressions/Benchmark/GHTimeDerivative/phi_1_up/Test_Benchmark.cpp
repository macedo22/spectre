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
#include "Executables/Benchmark/GHTimeDerivative/phi_1_up/BenchmarkedImpls.hpp"
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
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using phi_type = typename BenchmarkImpl::phi_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: inverse_spatial_metric
  const inverse_spatial_metric_type inverse_spatial_metric =
      make_with_random_values<inverse_spatial_metric_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: phi
  const phi_type phi = make_with_random_values<phi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // Compute manual result that returns LHS tensor
  const phi_1_up_type phi_1_up_manual_returned =
      BenchmarkImpl::manual_impl_lhs_return(inverse_spatial_metric, phi);

  // Compute TensorExpression result that returns LHS tensor
  const phi_1_up_type phi_1_up_te_returned =
      BenchmarkImpl::tensorexpression_impl_lhs_return(inverse_spatial_metric,
                                                      phi);

  // LHS: phi_1_up to be filled by manual impl
  phi_1_up_type phi_1_up_manual_filled(used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(make_not_null(&phi_1_up_manual_filled),
                                     inverse_spatial_metric, phi);

  // LHS: phi_1_up to be filled by TensorExpression impl<1>
  phi_1_up_type phi_1_up_te1_filled(used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&phi_1_up_te1_filled), inverse_spatial_metric, phi);

  // LHS: phi_1_up to be filled by TensorExpression impl<2>
  phi_1_up_type phi_1_up_te2_filled(used_for_size);

  // Compute TensorExpression impl<2> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<2>(
      make_not_null(&phi_1_up_te2_filled), inverse_spatial_metric, phi);

  // LHS: phi_1_up to be filled by TensorExpression impl<3>
  phi_1_up_type phi_1_up_te3_filled(used_for_size);

  // Compute TensorExpression impl<3> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<3>(
      make_not_null(&phi_1_up_te3_filled), inverse_spatial_metric, phi);

  // LHS: phi_1_up to be filled by TensorExpression impl<4>
  phi_1_up_type phi_1_up_te4_filled(used_for_size);

  // Compute TensorExpression impl<4> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<4>(
      make_not_null(&phi_1_up_te4_filled), inverse_spatial_metric, phi);

  // LHS: phi_1_up to be filled by TensorExpression impl<5>
  phi_1_up_type phi_1_up_te5_filled(used_for_size);

  // Compute TensorExpression impl<4> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<5>(
      make_not_null(&phi_1_up_te5_filled), inverse_spatial_metric, phi);

  // CHECK phi_1_up (iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_1_up_manual_returned.get(i, a, b),
                              phi_1_up_te_returned.get(i, a, b));
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te1_filled.get(i, a, b));
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te2_filled.get(i, a, b));
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te3_filled.get(i, a, b));
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te4_filled.get(i, a, b));
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te5_filled.get(i, a, b));
      }
    }
  }

  // === Check TE impl with TempTensors ===

  size_t num_grid_points = 0;
  if constexpr (std::is_same_v<DataType, DataVector>) {
    num_grid_points = used_for_size.size();
  }

  TempBuffer<tmpl::list<::Tags::TempTensor<0, phi_1_up_type>,
                        ::Tags::TempTensor<1, phi_1_up_type>,
                        ::Tags::TempTensor<2, phi_1_up_type>,
                        ::Tags::TempTensor<3, phi_1_up_type>,
                        ::Tags::TempTensor<4, phi_1_up_type>,
                        ::Tags::TempTensor<5, inverse_spatial_metric_type>,
                        ::Tags::TempTensor<6, phi_type>>>
      vars{num_grid_points};

  // RHS: inverse_spatial_metric
  inverse_spatial_metric_type& inverse_spatial_metric_te_temp =
      get<::Tags::TempTensor<5, inverse_spatial_metric_type>>(vars);
  copy_tensor(inverse_spatial_metric,
              make_not_null(&inverse_spatial_metric_te_temp));

  // RHS: phi
  phi_type& phi_te_temp = get<::Tags::TempTensor<6, phi_type>>(vars);
  copy_tensor(phi, make_not_null(&phi_te_temp));

  // LHS: phi_1_up impl<1>
  phi_1_up_type& phi_1_up_te1_temp =
      get<::Tags::TempTensor<0, phi_1_up_type>>(vars);

  // Compute TensorExpression impl<1> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&phi_1_up_te1_temp), inverse_spatial_metric_te_temp,
      phi_te_temp);

  // LHS: phi_1_up impl<2>
  phi_1_up_type& phi_1_up_te2_temp =
      get<::Tags::TempTensor<1, phi_1_up_type>>(vars);

  // Compute TensorExpression impl<2> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<2>(
      make_not_null(&phi_1_up_te2_temp), inverse_spatial_metric_te_temp,
      phi_te_temp);

  // LHS: phi_1_up impl<3>
  phi_1_up_type& phi_1_up_te3_temp =
      get<::Tags::TempTensor<2, phi_1_up_type>>(vars);

  // Compute TensorExpression impl<3> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<3>(
      make_not_null(&phi_1_up_te3_temp), inverse_spatial_metric_te_temp,
      phi_te_temp);

  // LHS: phi_1_up impl<4>
  phi_1_up_type& phi_1_up_te4_temp =
      get<::Tags::TempTensor<3, phi_1_up_type>>(vars);

  // Compute TensorExpression impl<4> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<4>(
      make_not_null(&phi_1_up_te4_temp), inverse_spatial_metric_te_temp,
      phi_te_temp);

  // LHS: phi_1_up impl<5>
  phi_1_up_type& phi_1_up_te5_temp =
      get<::Tags::TempTensor<4, phi_1_up_type>>(vars);

  // Compute TensorExpression impl<5> result
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<5>(
      make_not_null(&phi_1_up_te5_temp), inverse_spatial_metric_te_temp,
      phi_te_temp);

  // CHECK phi_1_up
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te1_temp.get(i, a, b));
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te2_temp.get(i, a, b));
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te3_temp.get(i, a, b));
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te4_temp.get(i, a, b));
        CHECK_ITERABLE_APPROX(phi_1_up_manual_filled.get(i, a, b),
                              phi_1_up_te5_temp.get(i, a, b));
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

SPECTRE_TEST_CASE("Unit.Benchmark.GHTimeDerivative.phi_1_up",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_impls(std::numeric_limits<double>::signaling_NaN(),
                         make_not_null(&generator));
  test_benchmarked_impls(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
