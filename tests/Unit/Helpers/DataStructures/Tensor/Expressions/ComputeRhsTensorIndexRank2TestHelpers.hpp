// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

// Check each element of each mapping
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

      CHECK(R_ab.template compute_rhs_tensor_index<2>(
                index_order_ab, index_order_ab, ij) == ij);
      CHECK(R_ab.template compute_rhs_tensor_index<2>(
                index_order_ba, index_order_ab, ij) == ji);
    }
  }
}

// Test all dimension combinations with grid and inertial frames
// for nonsymmetric indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
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

// Test all dimensions with grid and inertial frames
// for symmetric indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
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
