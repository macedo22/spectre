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
          typename TensorIndexA, typename TensorIndexB, typename TensorIndexC>
void test_storage_get_rank_3_core(const TensorIndexA& tensorindex_a,
                                  const TensorIndexB& tensorindex_b,
                                  const TensorIndexC& tensorindex_c,
                                  const size_t& spatial_dim_a,
                                  const size_t& spatial_dim_b,
                                  const size_t& spatial_dim_c) {
  Tensor<Datatype, Symmetry, TensorIndexTypeList> rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);

  auto abc_to_abc =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  auto abc_to_acb =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  auto abc_to_bac =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  auto abc_to_bca =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  auto abc_to_cab =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  auto abc_to_cba =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  for (size_t i = 0; i < spatial_dim_a; ++i) {
    for (size_t j = 0; j < spatial_dim_b; ++j) {
      for (size_t k = 0; k < spatial_dim_c; ++k) {
        CHECK(abc_to_abc.get(i, j, k) == rhs_tensor.get(i, j, k));
        CHECK(abc_to_acb.get(i, k, j) == rhs_tensor.get(i, j, k));
        CHECK(abc_to_bac.get(j, i, k) == rhs_tensor.get(i, j, k));
        CHECK(abc_to_bca.get(j, k, i) == rhs_tensor.get(i, j, k));
        CHECK(abc_to_cab.get(k, i, j) == rhs_tensor.get(i, j, k));
        CHECK(abc_to_cba.get(k, j, i) == rhs_tensor.get(i, j, k));

        CHECK(abc_to_abc.get(i, j, k) == abc_to_acb.get(i, k, j));
        CHECK(abc_to_abc.get(i, j, k) == abc_to_bac.get(j, i, k));
        CHECK(abc_to_abc.get(i, j, k) == abc_to_bca.get(j, k, i));
        CHECK(abc_to_abc.get(i, j, k) == abc_to_cab.get(k, i, j));
        CHECK(abc_to_abc.get(i, j, k) == abc_to_cba.get(k, j, i));
        CHECK(abc_to_acb.get(i, k, j) == abc_to_bac.get(j, i, k));
        CHECK(abc_to_acb.get(i, k, j) == abc_to_bca.get(j, k, i));
        CHECK(abc_to_acb.get(i, k, j) == abc_to_cab.get(k, i, j));
        CHECK(abc_to_acb.get(i, k, j) == abc_to_cba.get(k, j, i));
        CHECK(abc_to_bac.get(j, i, k) == abc_to_bca.get(j, k, i));
        CHECK(abc_to_bac.get(j, i, k) == abc_to_cab.get(k, i, j));
        CHECK(abc_to_bac.get(j, i, k) == abc_to_cba.get(k, j, i));
        CHECK(abc_to_bca.get(j, k, i) == abc_to_cab.get(k, i, j));
        CHECK(abc_to_bca.get(j, k, i) == abc_to_cba.get(k, j, i));
        CHECK(abc_to_cab.get(k, i, j) == abc_to_cba.get(k, j, i));
      }
    }
  }
}

// Test all dimension combinations with grid and inertial frames
// for nonsymmetric indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          typename TensorIndexC,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC>
void test_storage_get_rank_3_no_symmetry(const TensorIndexA& tensorindex_a,
                                         const TensorIndexB& tensorindex_b,
                                         const TensorIndexC& tensorindex_c) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(2, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(3, data)

#define CALL_TEST_STORAGE_GET_RANK_3_CORE(_, data)                            \
  test_storage_get_rank_3_core<                                               \
      Datatype, Symmetry<3, 2, 1>,                                            \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,        \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,        \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>,       \
      TensorIndexA, TensorIndexB, TensorIndexC>(tensorindex_a, tensorindex_b, \
                                                tensorindex_c, DIM_A(data),   \
                                                DIM_B(data), DIM_C(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_STORAGE_GET_RANK_3_CORE, (1, 2, 3),
                          (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_STORAGE_GET_RANK_3_CORE
#undef FRAME
#undef DIM_C
#undef DIM_B
#undef DIM_A
}

// Test all dimension combinations with grid and inertial frames
// for symmetric first and second indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          typename TensorIndexC,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC>
void test_storage_get_rank_3_ab_symmetry(const TensorIndexA& tensorindex_a,
                                         const TensorIndexB& tensorindex_b,
                                         const TensorIndexC& tensorindex_c) {
#define DIM_AB(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_STORAGE_GET_RANK_3_CORE(_, data)                            \
  test_storage_get_rank_3_core<                                               \
      Datatype, Symmetry<2, 2, 1>,                                            \
      index_list<TensorIndexTypeA<DIM_AB(data), ValenceA, FRAME(data)>,       \
                 TensorIndexTypeB<DIM_AB(data), ValenceB, FRAME(data)>,       \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>,       \
      TensorIndexA, TensorIndexB, TensorIndexC>(tensorindex_a, tensorindex_b, \
                                                tensorindex_c, DIM_AB(data),  \
                                                DIM_AB(data), DIM_C(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_STORAGE_GET_RANK_3_CORE, (1, 2, 3),
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_STORAGE_GET_RANK_3_CORE
#undef FRAME
#undef DIM_C
#undef DIM_AB
}

// Test all dimension combinations with grid and inertial frames
// for symmetric first and third indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          typename TensorIndexC,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC>
void test_storage_get_rank_3_ac_symmetry(const TensorIndexA& tensorindex_a,
                                         const TensorIndexB& tensorindex_b,
                                         const TensorIndexC& tensorindex_c) {
#define DIM_AC(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_STORAGE_GET_RANK_3_CORE(_, data)                            \
  test_storage_get_rank_3_core<                                               \
      Datatype, Symmetry<2, 1, 2>,                                            \
      index_list<TensorIndexTypeA<DIM_AC(data), ValenceA, FRAME(data)>,       \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,        \
                 TensorIndexTypeC<DIM_AC(data), ValenceC, FRAME(data)>>,      \
      TensorIndexA, TensorIndexB, TensorIndexC>(tensorindex_a, tensorindex_b, \
                                                tensorindex_c, DIM_AC(data),  \
                                                DIM_B(data), DIM_AC(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_STORAGE_GET_RANK_3_CORE, (1, 2, 3),
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_STORAGE_GET_RANK_3_CORE
#undef FRAME
#undef DIM_B
#undef DIM_AC
}

// Test all dimension combinations with grid and inertial frames
// for symmetric second and third indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          typename TensorIndexC,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC>
void test_storage_get_rank_3_bc_symmetry(const TensorIndexA& tensorindex_a,
                                         const TensorIndexB& tensorindex_b,
                                         const TensorIndexC& tensorindex_c) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_BC(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_STORAGE_GET_RANK_3_CORE(_, data)                            \
  test_storage_get_rank_3_core<                                               \
      Datatype, Symmetry<2, 1, 1>,                                            \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,        \
                 TensorIndexTypeB<DIM_BC(data), ValenceB, FRAME(data)>,       \
                 TensorIndexTypeC<DIM_BC(data), ValenceC, FRAME(data)>>,      \
      TensorIndexA, TensorIndexB, TensorIndexC>(tensorindex_a, tensorindex_b, \
                                                tensorindex_c, DIM_A(data),   \
                                                DIM_BC(data), DIM_BC(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_STORAGE_GET_RANK_3_CORE, (1, 2, 3),
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_STORAGE_GET_RANK_3_CORE
#undef FRAME
#undef DIM_BC
#undef DIM_A
}

// Test all dimension combinations with grid and inertial frames
// for symmetric indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          typename TensorIndexC,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC>
void test_storage_get_rank_3_abc_symmetry(const TensorIndexA& tensorindex_a,
                                          const TensorIndexB& tensorindex_b,
                                          const TensorIndexC& tensorindex_c) {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_STORAGE_GET_RANK_3_CORE(_, data)                            \
  test_storage_get_rank_3_core<                                               \
      Datatype, Symmetry<1, 1, 1>,                                            \
      index_list<TensorIndexTypeA<DIM(data), ValenceA, FRAME(data)>,          \
                 TensorIndexTypeB<DIM(data), ValenceB, FRAME(data)>,          \
                 TensorIndexTypeC<DIM(data), ValenceC, FRAME(data)>>,         \
      TensorIndexA, TensorIndexB, TensorIndexC>(tensorindex_a, tensorindex_b, \
                                                tensorindex_c, DIM(data),     \
                                                DIM(data), DIM(data));

  GENERATE_INSTANTIATIONS(CALL_TEST_STORAGE_GET_RANK_3_CORE, (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_STORAGE_GET_RANK_3_CORE
#undef FRAME
#undef DIM
}
