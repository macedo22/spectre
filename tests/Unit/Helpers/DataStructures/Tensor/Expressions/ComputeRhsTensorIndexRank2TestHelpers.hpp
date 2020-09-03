// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TestingFrameworkGroup
/// \brief Test that the tensor multi-index of a rank 2 RHS Tensor is equivalent
/// to the LHS tensor multi-index, according to the order of their generic
/// indices
///
/// \details `TensorIndexA` and `TensorIndexB` can be any type of TensorIndex
/// and are not necessarily `ti_a_t` and `ti_b_t`. The "A" and "B" suffixes just
/// denote the ordering of the generic indices of the RHS tensor expression. In
/// the RHS tensor expression, it means `TensorIndexA` is the first index used
/// and `TensorIndexB` is the second index used.
///
/// If we consider the RHS tensor's generic indices to be (a, b), the possible
/// orderings of the LHS tensor's generic indices are: (a, b) and (b, a). For
/// each of these cases, this test checks that for each LHS component's tensor
/// multi-index, the equivalent RHS tensor multi-index is correctly computed.
///
/// \tparam Datatype the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam TensorIndexA the type of the first TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_a_t`
/// \tparam TensorIndexB the type of the second TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_B_t`
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
template <typename Datatype, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename TensorIndexA,
          typename TensorIndexB>
void test_compute_rhs_tensor_index_rank_2_core(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b) {
  const Tensor<Datatype, RhsSymmetry, RhsTensorIndexTypeList> rhs_tensor{};
  const auto R_ab = rhs_tensor(tensorindex_a, tensorindex_b);

  const size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;

  const std::array<size_t, 2> index_order_ab = {TensorIndexA::value,
                                                TensorIndexB::value};
  const std::array<size_t, 2> index_order_ba = {TensorIndexB::value,
                                                TensorIndexA::value};

  for (size_t i = 0; i < dim_a; i++) {
    for (size_t j = 0; j < dim_b; j++) {
      const std::array<size_t, 2> ij = {i, j};
      const std::array<size_t, 2> ji = {j, i};

      // For L_{ab} = R_{ab}, check that L_{ij} == R_{ij}
      CHECK(R_ab.template compute_rhs_tensor_index<2>(
                index_order_ab, index_order_ab, ij) == ij);
      // For L_{ba} = R_{ab}, check that L_{ij} == R_{ji}
      CHECK(R_ab.template compute_rhs_tensor_index<2>(
                index_order_ba, index_order_ab, ij) == ji);
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of computing the RHS tensor multi-index equivalent of
/// the LHS tensor multi-index with rank 2 Tensors on multiple Frame types and
/// dimension combinations for nonsymmetric indices
///
/// \details `TensorIndexA` and `TensorIndexB` can be any type of TensorIndex
/// and are not necessarily `ti_a_t` and `ti_b_t`. The "A" and "B" suffixes just
/// denote the ordering of the generic indices of the RHS tensor expression. In
/// the RHS tensor expression, it means `TensorIndexA` is the first index used
/// and `TensorIndexB` is the second index used.
///
/// \tparam Datatype the type of data being stored in the Tensors
/// \tparam TensorIndexTypeA the \ref SpacetimeIndex "TensorIndexType" of the
/// first index of the RHS Tensor
/// \tparam TensorIndexTypeB the \ref SpacetimeIndex "TensorIndexType" of the
/// second index of the RHS Tensor
/// \tparam TensorIndexA the type of the first TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_a_t`
/// \tparam TensorIndexB the type of the second TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_B_t`
/// \tparam ValenceA the valence of the first index used on the RHS of the
/// TensorExpression
/// \tparam ValenceB the valence of the second index used on the RHS of the
/// TensorExpression
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
template <
    typename Datatype, template <size_t, UpLo, typename> class TensorIndexTypeA,
    template <size_t, UpLo, typename> class TensorIndexTypeB, UpLo ValenceA,
    UpLo ValenceB, typename TensorIndexA, typename TensorIndexB>
void test_compute_rhs_tensor_index_rank_2_no_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE(_, data)          \
  test_compute_rhs_tensor_index_rank_2_core<                             \
      Datatype, Symmetry<2, 1>,                                          \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,   \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE,
                          (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE
#undef FRAME
#undef DIM_B
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of computing the RHS tensor multi-index equivalent of
/// the LHS tensor multi-index with rank 2 Tensors on multiple Frame types and
/// dimension combinations for symmetric indices
///
/// \details `TensorIndexA` and `TensorIndexB` can be any type of TensorIndex
/// and are not necessarily `ti_a_t` and `ti_b_t`. The "A" and "B" suffixes just
/// denote the ordering of the generic indices of the RHS tensor expression. In
/// the RHS tensor expression, it means `TensorIndexA` is the first index used
/// and `TensorIndexB` is the second index used.
///
/// \tparam Datatype the type of data being stored in the Tensors
/// \tparam TensorIndexTypeA the \ref SpacetimeIndex "TensorIndexType" of the
/// first index of the RHS Tensor
/// \tparam TensorIndexTypeB the \ref SpacetimeIndex "TensorIndexType" of the
/// second index of the RHS Tensor
/// \tparam TensorIndexA the type of the first TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_a_t`
/// \tparam TensorIndexB the type of the second TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_B_t`
/// \tparam ValenceA the valence of the first index used on the RHS of the
/// TensorExpression
/// \tparam ValenceB the valence of the second index used on the RHS of the
/// TensorExpression
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
template <
    typename Datatype, template <size_t, UpLo, typename> class TensorIndexTypeA,
    template <size_t, UpLo, typename> class TensorIndexTypeB, UpLo ValenceA,
    UpLo ValenceB, typename TensorIndexA, typename TensorIndexB>
void test_compute_rhs_tensor_index_rank_2_symmetric(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b) {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE(_, data)       \
  test_compute_rhs_tensor_index_rank_2_core<                          \
      Datatype, Symmetry<1, 1>,                                       \
      index_list<TensorIndexTypeA<DIM(data), ValenceA, FRAME(data)>,  \
                 TensorIndexTypeB<DIM(data), ValenceB, FRAME(data)>>, \
      TensorIndexA, TensorIndexB>(tensorindex_a, tensorindex_b);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE,
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE
#undef FRAME
#undef DIM
}
