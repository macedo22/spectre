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
          typename TensorIndexA, typename TensorIndexB>
void test_storage_get_rank_2_core(const TensorIndexA& tensorindex_a,
                                  const TensorIndexB& tensorindex_b,
                                  const size_t& spatial_dim_a,
                                  const size_t& spatial_dim_b) {
  Tensor<Datatype, Symmetry, TensorIndexTypeList> rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);

  auto ab_to_ab = TensorExpressions::evaluate<TensorIndexA, TensorIndexB>(
      rhs_tensor(tensorindex_a, tensorindex_b));

  auto ab_to_ba = TensorExpressions::evaluate<TensorIndexB, TensorIndexA>(
      rhs_tensor(tensorindex_a, tensorindex_b));

  for (size_t i = 0; i < spatial_dim_a; ++i) {
    for (size_t j = 0; j < spatial_dim_b; ++j) {
      CHECK(ab_to_ab.get(i, j) == rhs_tensor.get(i, j));
      CHECK(ab_to_ba.get(j, i) == rhs_tensor.get(i, j));
      CHECK(ab_to_ab.get(i, j) == ab_to_ba.get(j, i));
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
void test_storage_get_rank_2_no_symmetry(const TensorIndexA& tensorindex_a,
                                         const TensorIndexB& tensorindex_b) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_STORAGE_GET_RANK_2_CORE(_, data)                           \
  test_storage_get_rank_2_core<                                              \
      Datatype, Symmetry<2, 1>,                                              \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,       \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>>,      \
      TensorIndexA, TensorIndexB>(tensorindex_a, tensorindex_b, DIM_A(data), \
                                  DIM_B(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_STORAGE_GET_RANK_2_CORE, (1, 2, 3),
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_STORAGE_GET_RANK_2_CORE
#undef FRAME
#undef DIM_B
#undef DIM_A
}

// Test all dimensions with grid and inertial frames
// for symmetric indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          UpLo ValenceA, UpLo ValenceB>
void test_storage_get_rank_2_symmetric(const TensorIndexA& tensorindex_a,
                                       const TensorIndexB& tensorindex_b) {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_STORAGE_GET_RANK_2_CORE(_, data)                         \
  test_storage_get_rank_2_core<                                            \
      Datatype, Symmetry<1, 1>,                                            \
      index_list<TensorIndexTypeA<DIM(data), ValenceA, FRAME(data)>,       \
                 TensorIndexTypeB<DIM(data), ValenceB, FRAME(data)>>,      \
      TensorIndexA, TensorIndexB>(tensorindex_a, tensorindex_b, DIM(data), \
                                  DIM(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_STORAGE_GET_RANK_2_CORE, (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_STORAGE_GET_RANK_2_CORE
#undef FRAME
#undef DIM
}
