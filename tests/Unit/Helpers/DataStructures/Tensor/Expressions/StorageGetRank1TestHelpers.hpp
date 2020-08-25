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

// figure out how to not need to pass in spatial dims and tensor index types
template <typename Datatype, typename Symmetry, typename TensorIndexTypeList,
          typename TensorIndex>
void test_storage_get_rank_1_core(
    const TensorIndex& tensorindex, const size_t& spatial_dim) {
  Tensor<Datatype, Symmetry, TensorIndexTypeList> rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);

  auto lhs_tensor = TensorExpressions::evaluate<TensorIndex>(
      rhs_tensor(tensorindex));

  for (size_t i = 0; i < spatial_dim; ++i) {
    CHECK(lhs_tensor.get(i) == rhs_tensor.get(i));
  }
}

// Test all dimensions with grid and inertial frames
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndex,
          template <size_t, UpLo, typename> class TensorIndexType, UpLo Valence>
void test_storage_get_rank_1(const TensorIndex& tensorindex) {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_STORAGE_GET_RANK_1_CORE(_, data)                  \
  test_storage_get_rank_1_core<                                     \
      Datatype, Symmetry<1>,                                        \
      index_list<TensorIndexType<DIM(data), Valence, FRAME(data)>>, \
      TensorIndex>(tensorindex, DIM(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_STORAGE_GET_RANK_1_CORE,
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_STORAGE_GET_RANK_1_CORE
#undef FRAME
#undef DIM
}
