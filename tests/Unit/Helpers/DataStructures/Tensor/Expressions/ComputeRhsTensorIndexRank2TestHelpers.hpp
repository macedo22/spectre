// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

//#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

// Check each element of each mapping
template <typename Datatype, typename Symmetry, typename IndexList,
          typename TensorIndexTypeA, typename TensorIndexTypeB>
void test_compute_rhs_tensor_index_rank_2_core(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b, const size_t& spatial_dim_a,
    const size_t& spatial_dim_b) {
  Tensor<Datatype, Symmetry, IndexList> rhs_tensor{};

  auto rhs_tensor_expr = rhs_tensor(tensor_index_type_a, tensor_index_type_b);

  std::array<size_t, 2> index_order_ab = {TensorIndexTypeA::value,
                                          TensorIndexTypeB::value};
  std::array<size_t, 2> index_order_ba = {TensorIndexTypeB::value,
                                          TensorIndexTypeA::value};

  for (size_t i = 0; i < spatial_dim_a; i++) {
    for (size_t j = 0; j < spatial_dim_b; j++) {
      const std::array<size_t, 2> ij = {i, j};
      const std::array<size_t, 2> ji = {j, i};

      CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<2>(
                index_order_ab, index_order_ab, ij) == ij);
      CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<2>(
                index_order_ba, index_order_ab, ij) == ji);
    }
  }
}

// Test all dimension combinations with grid and inertial frames
// for nonsymmetric indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          UpLo ValenceA, UpLo ValenceB>
void test_compute_rhs_tensor_index_rank_2_no_symmetry(
    const TensorIndexA& tensor_index_a, const TensorIndexB& tensor_index_b) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE(_, data)                \
  test_compute_rhs_tensor_index_rank_2_core<                                   \
      Datatype, Symmetry<2, 1>,                                                \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,         \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>>,        \
      TensorIndexA, TensorIndexB>(tensor_index_a, tensor_index_b, DIM_A(data), \
                                  DIM_B(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE,
                          (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE
#undef FRAME
#undef DIM_B
#undef DIM_A
}

// Test all dimension combinations with grid and inertial frames
// for symmetric indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          UpLo ValenceA, UpLo ValenceB>
void test_compute_rhs_tensor_index_rank_2_symmetric(
    const TensorIndexA& tensor_index_a, const TensorIndexB& tensor_index_b) {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE(_, data)              \
  test_compute_rhs_tensor_index_rank_2_core<                                 \
      Datatype, Symmetry<1, 1>,                                              \
      index_list<TensorIndexTypeA<DIM(data), ValenceA, FRAME(data)>,         \
                 TensorIndexTypeB<DIM(data), ValenceB, FRAME(data)>>,        \
      TensorIndexA, TensorIndexB>(tensor_index_a, tensor_index_b, DIM(data), \
                                  DIM(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE,
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_2_CORE
#undef FRAME
#undef DIM
}
