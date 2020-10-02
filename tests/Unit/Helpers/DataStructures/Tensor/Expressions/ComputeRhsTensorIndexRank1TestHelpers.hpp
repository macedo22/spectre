// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::TensorExpressions {

/// \ingroup TestingFrameworkGroup
/// \brief Test that the computed tensor multi-index of a rank 1 RHS Tensor is
/// equivalent to the given LHS tensor multi-index
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam TensorIndexTypeList the Tensors' typelist containing their
/// \ref SpacetimeIndex "TensorIndexType"
/// \param tensorindex the TensorIndex used in the the TensorExpression,
/// e.g. `ti_a`
template <typename DataType, typename TensorIndexTypeList, typename TensorIndex>
void test_compute_rhs_tensor_index_rank_1_core(
    const TensorIndex& tensorindex) noexcept {
  const Tensor<DataType, Symmetry<1>, TensorIndexTypeList> rhs_tensor(5_st);
  const auto R_a = rhs_tensor(tensorindex);

  const size_t dim = tmpl::at_c<TensorIndexTypeList, 0>::dim;

  const std::array<size_t, 1> index_order = {TensorIndex::value};

  // For L_a = R_a, check that L_i == R_i
  for (size_t i = 0; i < dim; i++) {
    const std::array<size_t, 1> tensor_multi_index = {i};
    CHECK(R_a.template compute_rhs_tensor_index<1>(index_order, index_order,
                                                   tensor_multi_index) ==
          tensor_multi_index);
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of computing the RHS tensor multi-index equivalent of
/// the LHS tensor multi-index with rank 1 Tensors on multiple Frame types and
/// dimensions
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam TensorIndexType the Tensors' \ref SpacetimeIndex "TensorIndexType"
/// \tparam Valence the valence of the Tensors' index
/// \param tensorindex the TensorIndex used in the the TensorExpression,
/// e.g. `ti_a`
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexType, UpLo Valence,
          typename TensorIndex>
void test_compute_rhs_tensor_index_rank_1(
    const TensorIndex& tensorindex) noexcept {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_1_CORE(_, data)                \
  test_compute_rhs_tensor_index_rank_1_core<                                   \
      DataType, index_list<TensorIndexType<DIM(data), Valence, FRAME(data)>>>( \
      tensorindex);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_1_CORE,
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_1_CORE
#undef FRAME
#undef DIM
}

}  // namespace TestHelpers::TensorExpressions
