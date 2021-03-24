// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Executables/Benchmark/BenchmarkedImpls.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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

// Make sure TE impl matches manual impl
template <size_t Dim, typename DataType>
void test_benchmarked_implementations_core(
    const DataType& used_for_size) noexcept {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using L_type = typename BenchmarkImpl::L_type;
  using R_type = typename BenchmarkImpl::R_type;
  using S_type = typename BenchmarkImpl::S_type;
  using T_type = typename BenchmarkImpl::T_type;

  // R
  R_type R(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&R));

  // S
  S_type S(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&S));

  // T
  T_type T(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&T));

  // LHS: L for manual implementation
  const L_type L_manual =
      BenchmarkImpl::manual_impl_return(R, S, T, used_for_size);

  // LHS: L for TE implementation with LHS returned
  const L_type L_te_returned =
      BenchmarkImpl::tensorexpression_impl_return(R, S, T);

  // LHS: L for TE implementation with LHS pass as arg
  L_type L_te_filled(used_for_size);
  BenchmarkImpl::tensorexpression_impl_lhs_as_arg(make_not_null(&L_te_filled),
                                                  R, S, T);

  for (size_t c = 0; c < tmpl::at_c<typename L_type::index_list, 0>::dim; c++) {
    for (size_t d = 0; d < tmpl::at_c<typename L_type::index_list, 1>::dim;
         d++) {
      for (size_t k = 0; k < tmpl::at_c<typename L_type::index_list, 2>::dim;
           k++) {
        for (size_t l = 0; l < tmpl::at_c<typename L_type::index_list, 3>::dim;
             l++) {
          CHECK_ITERABLE_APPROX(L_manual.get(c, d, k, l),
                                L_te_returned.get(c, d, k, l));
          CHECK_ITERABLE_APPROX(L_manual.get(c, d, k, l),
                                L_te_filled.get(c, d, k, l));
        }
      }
    }
  }

  if constexpr (not std::is_same_v<DataType, double>) {
    TempBuffer<tmpl::list<
        ::Tags::TempTensor<0, L_type>, ::Tags::TempTensor<1, R_type>,
        ::Tags::TempTensor<2, S_type>, ::Tags::TempTensor<3, T_type>>>
        vars{used_for_size.size()};

    // R
    R_type& R_te_temp = get<::Tags::TempTensor<1, R_type>>(vars);
    assign_unique_values_to_tensor(make_not_null(&R_te_temp));

    // S
    S_type& S_te_temp = get<::Tags::TempTensor<2, S_type>>(vars);
    assign_unique_values_to_tensor(make_not_null(&S_te_temp));

    // T
    T_type& T_te_temp = get<::Tags::TempTensor<3, T_type>>(vars);
    assign_unique_values_to_tensor(make_not_null(&T_te_temp));

    // LHS: L
    L_type& L_te_temp = get<::Tags::TempTensor<0, L_type>>(vars);
    BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
        make_not_null(&L_te_temp), R_te_temp, S_te_temp, T_te_temp);

    for (size_t c = 0; c < tmpl::at_c<typename L_type::index_list, 0>::dim;
         c++) {
      for (size_t d = 0; d < tmpl::at_c<typename L_type::index_list, 1>::dim;
           d++) {
        for (size_t k = 0; k < tmpl::at_c<typename L_type::index_list, 2>::dim;
             k++) {
          for (size_t l = 0;
               l < tmpl::at_c<typename L_type::index_list, 3>::dim; l++) {
            CHECK_ITERABLE_APPROX(L_manual.get(c, d, k, l),
                                  L_te_temp.get(c, d, k, l));
          }
        }
      }
    }
  }
}

template <typename DataType>
void test_benchmarked_implementations(const DataType& used_for_size) noexcept {
  test_benchmarked_implementations_core<1>(used_for_size);
  test_benchmarked_implementations_core<2>(used_for_size);
  test_benchmarked_implementations_core<3>(used_for_size);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Benchmark",
                  "[DataStructures][Unit]") {
  test_benchmarked_implementations(
      std::numeric_limits<double>::signaling_NaN());
  test_benchmarked_implementations(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
