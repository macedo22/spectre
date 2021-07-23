// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename... Ts>
void assign_unique_values_to_tensor(
    const gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    const gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

// Computes \f$L_{a} = R_{ab} * S^{b} + G_{a} - H_{ba}{}^{b} * T\f$
template <typename R_t, typename S_t, typename G_t, typename H_t, typename T_t,
          typename DataType>
G_t compute_expected_result1(const R_t& R, const S_t& S, const G_t& G,
                             const H_t& H, const T_t& T,
                             const DataType& used_for_size) noexcept {
  using result_tensor_type = G_t;
  result_tensor_type expected_result{};
  const size_t dim = tmpl::front<typename R_t::index_list>::dim;
  for (size_t a = 0; a < dim; a++) {
    DataType expected_Rab_SB_product =
        make_with_value<DataType>(used_for_size, 0.0);
    DataType expected_HbaB_contracted_value =
        make_with_value<DataType>(used_for_size, 0.0);
    for (size_t b = 0; b < dim; b++) {
      expected_Rab_SB_product += R.get(a, b) * S.get(b);
      expected_HbaB_contracted_value += H.get(b, a, b);
    }
    expected_result.get(a) = expected_Rab_SB_product + G.get(a) -
                             (expected_HbaB_contracted_value * T.get());
  }

  return expected_result;
}

// Computes \f$N = \sqrt{g^{ij} * \psi_{jt} * \psi_{it} - \psi_{tt}}\f$
// i.e. Computes the lapse from the inverse spatial metric and spacetime metric
template <typename DataType>
Scalar<DataType> compute_expected_result2(
    const tnsr::II<DataType, 3, Frame::Inertial>& g,
    const tnsr::aa<DataType, 3, Frame::Inertial>& psi,
    const DataType& used_for_size) noexcept {
  DataType expected_g_psi_psi_product =
      make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      expected_g_psi_psi_product +=
          g.get(i, j) * psi.get(j + 1, 0) * psi.get(i + 1, 0);
    }
  }

  Scalar<DataType> expected_result{
      sqrt(expected_g_psi_psi_product - psi.get(0, 0))};

  return expected_result;
}

// Includes an expression with addition, subtraction, an inner product, an outer
// product, a contraction, and a scalar
template <typename DataType, typename Generator>
void test_case1(const DataType& used_for_size,
                const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto R = make_with_random_values<tnsr::ab<DataType, 3, Frame::Grid>>(
      generator, make_not_null(&distribution), used_for_size);

  const auto S = make_with_random_values<tnsr::A<DataType, 3, Frame::Grid>>(
      generator, make_not_null(&distribution), used_for_size);

  const auto G = make_with_random_values<tnsr::a<DataType, 3, Frame::Grid>>(
      generator, make_not_null(&distribution), used_for_size);

  const auto H = make_with_random_values<tnsr::abC<DataType, 3, Frame::Grid>>(
      generator, make_not_null(&distribution), used_for_size);

  const auto T = make_with_random_values<Scalar<DataType>>(
      generator, make_not_null(&distribution), used_for_size);

  using result_tensor_type = tnsr::a<DataType, 3, Frame::Grid>;
  result_tensor_type expected_result_tensor =
      compute_expected_result1(R, S, G, H, T, used_for_size);
  // \f$L_{a} = R_{ab}* S^{b} + G_{a} - H_{ba}{}^{b} * T\f$
  result_tensor_type actual_result_tensor_returned =
      TensorExpressions::evaluate<ti_a>(R(ti_a, ti_b) * S(ti_B) + G(ti_a) -
                                        H(ti_b, ti_a, ti_B) * T());
  result_tensor_type actual_result_tensor_filled{};
  TensorExpressions::evaluate<ti_a>(
      make_not_null(&actual_result_tensor_filled),
      R(ti_a, ti_b) * S(ti_B) + G(ti_a) - H(ti_b, ti_a, ti_B) * T());

  for (size_t a = 0; a < 4; a++) {
    CHECK_ITERABLE_APPROX(actual_result_tensor_returned.get(a),
                          expected_result_tensor.get(a));
    CHECK_ITERABLE_APPROX(actual_result_tensor_filled.get(a),
                          expected_result_tensor.get(a));
  }

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    Variables<tmpl::list<::Tags::TempTensor<1, result_tensor_type>>>
        actual_result_tensor_temp_var{used_for_size.size()};
    result_tensor_type& actual_result_tensor_temp =
        get<::Tags::TempTensor<1, result_tensor_type>>(
            actual_result_tensor_temp_var);
    ::TensorExpressions::evaluate<ti_a>(
        make_not_null(&actual_result_tensor_temp),
        R(ti_a, ti_b) * S(ti_B) + G(ti_a) - H(ti_b, ti_a, ti_B) * T());

    for (size_t a = 0; a < 4; a++) {
      CHECK_ITERABLE_APPROX(actual_result_tensor_temp.get(a),
                            expected_result_tensor.get(a));
    }
  }
}

// Includes an expression with subtraction, inner products, an outer product,
// a square root, a scalar, generic spatial indices used for spacetime indices,
// and concrete time indices used for spacetime indices
//
// Note: This is an expanded calculation of the lapse from the shift and
// spacetime metric, where the shift is calculated from the inverse spatial
// metric and spacetime metric.
template <typename DataType, typename Generator>
void test_case2(const DataType& used_for_size,
                const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(1.0, 2.0);

  const auto g =
      make_with_random_values<tnsr::II<DataType, 3, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  tnsr::aa<DataType, 3, Frame::Inertial> psi(used_for_size);
  gr::spacetime_metric(
      make_not_null(&psi),
      TestHelpers::gr::random_lapse(generator, used_for_size),
      TestHelpers::gr::random_shift<3>(generator, used_for_size),
      TestHelpers::gr::random_spatial_metric<3>(generator, used_for_size));

  const Scalar<DataType> expected_result_tensor =
      compute_expected_result2(g, psi, used_for_size);
  // \f$N = \sqrt{g^{ij} * \psi_{jt} * \psi_{it} - \psi_{tt}}\f$
  const Scalar<DataType> actual_result_tensor_returned =
      TensorExpressions::evaluate(sqrt(
          g(ti_I, ti_J) * psi(ti_j, ti_t) * psi(ti_i, ti_t) - psi(ti_t, ti_t)));
  Scalar<DataType> actual_result_tensor_filled{};
  TensorExpressions::evaluate(
      make_not_null(&actual_result_tensor_filled),
      sqrt(g(ti_I, ti_J) * psi(ti_j, ti_t) * psi(ti_i, ti_t) -
           psi(ti_t, ti_t)));

  CHECK_ITERABLE_APPROX(actual_result_tensor_returned.get(),
                        expected_result_tensor.get());

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    Variables<tmpl::list<::Tags::TempTensor<1, Tensor<DataType>>>>
        actual_result_tensor_temp_var{used_for_size.size()};
    Scalar<DataType>& actual_result_tensor_temp =
        get<::Tags::TempTensor<1, Tensor<DataType>>>(
            actual_result_tensor_temp_var);
    ::TensorExpressions::evaluate(
        make_not_null(&actual_result_tensor_temp),
        sqrt(g(ti_I, ti_J) * psi(ti_j, ti_t) * psi(ti_i, ti_t) -
             psi(ti_t, ti_t)));

    CHECK_ITERABLE_APPROX(actual_result_tensor_temp.get(),
                          expected_result_tensor.get());
  }
}

template <typename DataType, typename Generator>
void test_mixed_operations(const DataType& used_for_size,
                           const gsl::not_null<Generator*> generator) noexcept {
  test_case1(used_for_size, generator);
  test_case2(used_for_size, generator);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.MixedOperations",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator, 3);

  test_mixed_operations(std::numeric_limits<double>::signaling_NaN(),
                        make_not_null(&generator));
  test_mixed_operations(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
}
