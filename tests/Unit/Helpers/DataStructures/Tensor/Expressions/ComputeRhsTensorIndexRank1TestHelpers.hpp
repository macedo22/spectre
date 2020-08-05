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
#include "Utilities/TMPL.hpp"

// Check mapping for all positions from 0 to spatial_dim
template <typename Datatype, typename Symmetry, typename IndexList,
          typename TensorIndexType>
void test_compute_rhs_tensor_index_rank_1_core(
    const TensorIndexType& tensor_index_type, const size_t& spatial_dim) {
  Tensor<Datatype, Symmetry, IndexList> rhs_tensor{};

  auto rhs_tensor_expr = rhs_tensor(tensor_index_type);

  std::array<size_t, 1> index_order = {TensorIndexType::value};

  for (size_t i = 0; i < spatial_dim; i++) {
      const std::array<size_t, 1> i_arr = {i};
      CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<1>(
                index_order, index_order, i_arr) == i_arr);
  }
}

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexType,
          template <size_t, UpLo, typename> class IndexType, UpLo Valence>
void test_compute_rhs_tensor_index_rank_1(
    const TensorIndexType& tensor_index_type) {
  // Testing all dimensions with grid frame
  test_compute_rhs_tensor_index_rank_1_core<
      Datatype,
      Symmetry<1>,
      index_list<IndexType<1, Valence, Frame::Grid>>,
      TensorIndexType>(tensor_index_type, 1);

  test_compute_rhs_tensor_index_rank_1_core<
      Datatype,
      Symmetry<1>,
      index_list<IndexType<2, Valence, Frame::Grid>>,
      TensorIndexType>(tensor_index_type, 2);

  test_compute_rhs_tensor_index_rank_1_core<
      Datatype,
      Symmetry<1>,
      index_list<IndexType<3, Valence, Frame::Grid>>,
      TensorIndexType>(tensor_index_type, 3);

  // Testing all dimensions with inertial frame
  test_compute_rhs_tensor_index_rank_1_core<
      Datatype,
      Symmetry<1>,
      index_list<IndexType<1, Valence, Frame::Inertial>>,
      TensorIndexType>(tensor_index_type, 1);

  test_compute_rhs_tensor_index_rank_1_core<
      Datatype,
      Symmetry<1>,
      index_list<IndexType<2, Valence, Frame::Inertial>>,
      TensorIndexType>(tensor_index_type, 2);

  test_compute_rhs_tensor_index_rank_1_core<
      Datatype,
      Symmetry<1>,
      index_list<IndexType<3, Valence, Frame::Inertial>>,
      TensorIndexType>(tensor_index_type, 3);
}
