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
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using phi_type = typename BenchmarkImpl::phi_type;

  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: gamma1_plus_1
  const gamma1_plus_1_type gamma1_plus_1 =
      make_with_random_values<gamma1_plus_1_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: shift_dot_three_index_constraint
  const shift_dot_three_index_constraint_type shift_dot_three_index_constraint =
      make_with_random_values<shift_dot_three_index_constraint_type>(
          generator, make_not_null(&distribution), used_for_size);

  // RHS: shift
  const shift_type shift = make_with_random_values<shift_type>(
      generator, make_not_null(&distribution), used_for_size);

  // RHS: phi
  const phi_type phi = make_with_random_values<phi_type>(
      generator, make_not_null(&distribution), used_for_size);

  // Compute manual result that returns LHS tensor
  const dt_spacetime_metric_type dt_spacetime_metric_manual_returned =
      BenchmarkImpl::manual_impl_lhs_return(lapse, pi, gamma1_plus_1,
                                            shift_dot_three_index_constraint,
                                            shift, phi);

  // Compute TensorExpression result that returns LHS tensor
  const dt_spacetime_metric_type dt_spacetime_metric_te_returned =
      BenchmarkImpl::tensorexpression_impl_lhs_return(
          lapse, pi, gamma1_plus_1, shift_dot_three_index_constraint, shift,
          phi);

  // LHS: dt_spacetime_metric to be filled by manual impl
  dt_spacetime_metric_type dt_spacetime_metric_manual_filled(used_for_size);

  // Compute manual result with LHS tensor as argument
  BenchmarkImpl::manual_impl_lhs_arg(
      make_not_null(&dt_spacetime_metric_manual_filled), lapse, pi,
      gamma1_plus_1, shift_dot_three_index_constraint, shift, phi);

  // LHS: dt_spacetime_metric to be filled by TensorExpression impl<1>
  dt_spacetime_metric_type dt_spacetime_metric_te1_filled(used_for_size);

  // Compute TensorExpression impl<1> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
      make_not_null(&dt_spacetime_metric_te1_filled), lapse, pi, gamma1_plus_1,
      shift_dot_three_index_constraint, shift, phi);

  // LHS: dt_spacetime_metric to be filled by TensorExpression impl<2>
  dt_spacetime_metric_type dt_spacetime_metric_te2_filled(used_for_size);

  // Compute TensorExpression impl<2> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<2>(
      make_not_null(&dt_spacetime_metric_te2_filled), lapse, pi, gamma1_plus_1,
      shift_dot_three_index_constraint, shift, phi);

  // LHS: dt_spacetime_metric to be filled by TensorExpression impl<3>
  dt_spacetime_metric_type dt_spacetime_metric_te3_filled(used_for_size);

  // Compute TensorExpression impl<3> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<3>(
      make_not_null(&dt_spacetime_metric_te3_filled), lapse, pi, gamma1_plus_1,
      shift_dot_three_index_constraint, shift, phi);

  // LHS: dt_spacetime_metric to be filled by TensorExpression impl<4>
  dt_spacetime_metric_type dt_spacetime_metric_te4_filled(used_for_size);

  // Compute TensorExpression impl<4> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<4>(
      make_not_null(&dt_spacetime_metric_te4_filled), lapse, pi, gamma1_plus_1,
      shift_dot_three_index_constraint, shift, phi);

  // LHS: dt_spacetime_metric to be filled by TensorExpression impl<5>
  dt_spacetime_metric_type dt_spacetime_metric_te5_filled(used_for_size);

  // Compute TensorExpression impl<5> result with LHS tensor as argument
  BenchmarkImpl::template tensorexpression_impl_lhs_arg<5>(
      make_not_null(&dt_spacetime_metric_te5_filled), lapse, pi, gamma1_plus_1,
      shift_dot_three_index_constraint, shift, phi);

  // CHECK dt_spacetime_metric
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_returned.get(a, b),
                            dt_spacetime_metric_te_returned.get(a, b));
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                            dt_spacetime_metric_te1_filled.get(a, b));
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                            dt_spacetime_metric_te2_filled.get(a, b));
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                            dt_spacetime_metric_te3_filled.get(a, b));
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                            dt_spacetime_metric_te4_filled.get(a, b));
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                            dt_spacetime_metric_te5_filled.get(a, b));
    }
  }

  // For DataVectors, check TE impl with TempTensors
  if constexpr (not std::is_same_v<DataType, double>) {
    TempBuffer<tmpl::list<
        ::Tags::TempTensor<0, dt_spacetime_metric_type>,
        ::Tags::TempTensor<1, dt_spacetime_metric_type>,
        ::Tags::TempTensor<2, dt_spacetime_metric_type>,
        ::Tags::TempTensor<3, dt_spacetime_metric_type>,
        ::Tags::TempTensor<4, dt_spacetime_metric_type>,
        ::Tags::TempTensor<5, lapse_type>, ::Tags::TempTensor<6, pi_type>,
        ::Tags::TempTensor<7, gamma1_plus_1_type>,
        ::Tags::TempTensor<8, shift_dot_three_index_constraint_type>,
        ::Tags::TempTensor<9, shift_type>, ::Tags::TempTensor<10, phi_type>>>
        vars{used_for_size.size()};

    // RHS: lapse
    lapse_type& lapse_te_temp = get<::Tags::TempTensor<5, lapse_type>>(vars);
    copy_tensor(lapse, make_not_null(&lapse_te_temp));

    // RHS: pi
    pi_type& pi_te_temp = get<::Tags::TempTensor<6, pi_type>>(vars);
    copy_tensor(pi, make_not_null(&pi_te_temp));

    // RHS: gamma1_plus_1
    gamma1_plus_1_type& gamma1_plus_1_te_temp =
        get<::Tags::TempTensor<7, gamma1_plus_1_type>>(vars);
    copy_tensor(gamma1_plus_1, make_not_null(&gamma1_plus_1_te_temp));

    // RHS: shift_dot_three_index_constraint
    shift_dot_three_index_constraint_type&
        shift_dot_three_index_constraint_te_temp =
            get<::Tags::TempTensor<8, shift_dot_three_index_constraint_type>>(
                vars);
    copy_tensor(shift_dot_three_index_constraint,
                make_not_null(&shift_dot_three_index_constraint_te_temp));

    // RHS: shift
    shift_type& shift_te_temp = get<::Tags::TempTensor<9, shift_type>>(vars);
    copy_tensor(shift, make_not_null(&shift_te_temp));

    // RHS: phi
    phi_type& phi_te_temp = get<::Tags::TempTensor<10, phi_type>>(vars);
    copy_tensor(phi, make_not_null(&phi_te_temp));

    // LHS: dt_spacetime_metric impl<1>
    dt_spacetime_metric_type& dt_spacetime_metric_te1_temp =
        get<::Tags::TempTensor<0, dt_spacetime_metric_type>>(vars);

    // Compute TensorExpression impl<1> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<1>(
        make_not_null(&dt_spacetime_metric_te1_temp), lapse, pi, gamma1_plus_1,
        shift_dot_three_index_constraint, shift, phi);

    // LHS: dt_spacetime_metric impl<2>
    dt_spacetime_metric_type& dt_spacetime_metric_te2_temp =
        get<::Tags::TempTensor<1, dt_spacetime_metric_type>>(vars);

    // Compute TensorExpression impl<2> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<2>(
        make_not_null(&dt_spacetime_metric_te2_temp), lapse, pi, gamma1_plus_1,
        shift_dot_three_index_constraint, shift, phi);

    // LHS: dt_spacetime_metric impl<3>
    dt_spacetime_metric_type& dt_spacetime_metric_te3_temp =
        get<::Tags::TempTensor<2, dt_spacetime_metric_type>>(vars);

    // Compute TensorExpression impl<3> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<3>(
        make_not_null(&dt_spacetime_metric_te3_temp), lapse, pi, gamma1_plus_1,
        shift_dot_three_index_constraint, shift, phi);

    // LHS: dt_spacetime_metric impl<4>
    dt_spacetime_metric_type& dt_spacetime_metric_te4_temp =
        get<::Tags::TempTensor<3, dt_spacetime_metric_type>>(vars);

    // Compute TensorExpression impl<4> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<4>(
        make_not_null(&dt_spacetime_metric_te4_temp), lapse, pi, gamma1_plus_1,
        shift_dot_three_index_constraint, shift, phi);

    // LHS: dt_spacetime_metric impl<5>
    dt_spacetime_metric_type& dt_spacetime_metric_te5_temp =
        get<::Tags::TempTensor<4, dt_spacetime_metric_type>>(vars);

    // Compute TensorExpression impl<5> result
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<5>(
        make_not_null(&dt_spacetime_metric_te5_temp), lapse, pi, gamma1_plus_1,
        shift_dot_three_index_constraint, shift, phi);

    // CHECK dt_spacetime_metric
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                              dt_spacetime_metric_te1_temp.get(a, b));
        CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                              dt_spacetime_metric_te2_temp.get(a, b));
        CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                              dt_spacetime_metric_te3_temp.get(a, b));
        CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                              dt_spacetime_metric_te4_temp.get(a, b));
        CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual_filled.get(a, b),
                              dt_spacetime_metric_te5_temp.get(a, b));
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
