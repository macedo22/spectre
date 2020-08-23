// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

// figure out how to note need to pass in spatial dims and tensor index types
template <typename Datatype, typename Symmetry, typename TensorIndexTypeList,
          typename TensorIndexA, typename TensorIndexB, typename TensorIndexC,
          typename TensorIndexD>
void test_storage_get_rank_4(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c, const TensorIndexD& tensorindex_d,
    const size_t& spatial_dim_a, const size_t& spatial_dim_b,
    const size_t& spatial_dim_c, const size_t& spatial_dim_d) {
  Tensor<Datatype, Symmetry, TensorIndexTypeList> rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);

  auto abcd_to_abcd =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_abdc =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexD,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_acbd =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexB,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_acdb =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexD,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_adbc =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexD, TensorIndexB,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_adcb =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexD, TensorIndexC,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_bacd =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexC,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_badc =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexD,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_bcad =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexA,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_bcda =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexD,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_bdac =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexD, TensorIndexA,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_bdca =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexD, TensorIndexC,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_cabd =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexB,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_cadb =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexD,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_cbad =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexA,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_cbda =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexD,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_cdab =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexD, TensorIndexA,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_cdba =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexD, TensorIndexB,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_dabc =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexA, TensorIndexB,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_dacb =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexA, TensorIndexC,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_dbac =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexB, TensorIndexA,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_dbca =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexB, TensorIndexC,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_dcab =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexC, TensorIndexA,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  auto abcd_to_dcba =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexC, TensorIndexB,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  for (size_t i = 0; i < spatial_dim_a; ++i) {
    for (size_t j = 0; j < spatial_dim_b; ++j) {
      for (size_t k = 0; k < spatial_dim_c; ++k) {
          for (size_t l = 0; l < spatial_dim_d; ++l) {
            CHECK(abcd_to_abcd.get(i, j, k, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_abdc.get(i, j, l, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_acbd.get(i, k, j, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_acdb.get(i, k, l, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_adbc.get(i, l, j, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_adcb.get(i, l, k, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_bacd.get(j, i, k, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_badc.get(j, i, l, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_bcad.get(j, k, i, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_bcda.get(j, k, l, i) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_bdac.get(j, l, i, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_bdca.get(j, l, k, i) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cabd.get(k, i, j, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cadb.get(k, i, l, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cbad.get(k, j, i, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cbda.get(k, j, l, i) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cdab.get(k, l, i, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cdba.get(k, l, j, i) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dabc.get(l, i, j, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dacb.get(l, i, k, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dbac.get(l, j, i, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dbca.get(l, j, k, i) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dcab.get(l, k, i, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dcba.get(l, k, j, i) == rhs_tensor.get(i, j, k, l));
        }
      }
    }
  }
}
