// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

// Check each element of each mapping
template <typename Datatype, typename Symmetry, typename TensorIndexTypeList,
          typename TensorIndex>
void test_compute_rhs_tensor_index_rank_1_core(const TensorIndex& tensorindex,
                                               const size_t& spatial_dim) {
  Tensor<Datatype, Symmetry, TensorIndexTypeList> rhs_tensor{};

  auto rhs_tensor_expr = rhs_tensor(tensorindex);

  std::array<size_t, 1> index_order = {TensorIndex::value};

  for (size_t i = 0; i < spatial_dim; i++) {
      const std::array<size_t, 1> i_arr = {i};
      CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<1>(
                index_order, index_order, i_arr) == i_arr);
  }
}

// Test all dimensions with grid and inertial frames
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndex,
          template <size_t, UpLo, typename> class TensorIndexType, UpLo Valence>
void test_compute_rhs_tensor_index_rank_1(const TensorIndex& tensorindex) {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_1_CORE(_, data)     \
  test_compute_rhs_tensor_index_rank_1_core<                        \
      Datatype, Symmetry<1>,                                        \
      index_list<TensorIndexType<DIM(data), Valence, FRAME(data)>>, \
      TensorIndex>(tensorindex, DIM(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_1_CORE,
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_1_CORE
#undef FRAME
#undef DIM
}
