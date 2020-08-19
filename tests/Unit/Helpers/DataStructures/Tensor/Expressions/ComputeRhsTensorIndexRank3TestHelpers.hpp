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

// *** The last (outermost) one should go through all
// grid types and spatial dimensions
template <typename Datatype, typename Symmetry, typename IndexList,
          typename TensorIndexTypeA, typename TensorIndexTypeB,
          typename TensorIndexTypeC>
void test_compute_rhs_tensor_index_rank_3_core_impl(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c, const size_t& spatial_dim_a,
    const size_t& spatial_dim_b, const size_t& spatial_dim_c) {
  Tensor<Datatype, Symmetry, IndexList> rhs_tensor{};

  auto rhs_tensor_expr =
      rhs_tensor(tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  std::array<size_t, 3> index_order_abc = {TensorIndexTypeA::value,
                                           TensorIndexTypeB::value,
                                           TensorIndexTypeC::value};
  std::array<size_t, 3> index_order_acb = {TensorIndexTypeA::value,
                                           TensorIndexTypeC::value,
                                           TensorIndexTypeB::value};
  std::array<size_t, 3> index_order_bac = {TensorIndexTypeB::value,
                                           TensorIndexTypeA::value,
                                           TensorIndexTypeC::value};
  std::array<size_t, 3> index_order_bca = {TensorIndexTypeB::value,
                                           TensorIndexTypeC::value,
                                           TensorIndexTypeA::value};
  std::array<size_t, 3> index_order_cab = {TensorIndexTypeC::value,
                                           TensorIndexTypeA::value,
                                           TensorIndexTypeB::value};
  std::array<size_t, 3> index_order_cba = {TensorIndexTypeC::value,
                                           TensorIndexTypeB::value,
                                           TensorIndexTypeA::value};


  for (size_t i = 0; i < spatial_dim_a; i++) {
    for (size_t j = 0; j < spatial_dim_b; j++) {
      for (size_t k = 0; k < spatial_dim_c; k++) {
        std::array<size_t, 3> ijk = {i, j, k};
        std::array<size_t, 3> ikj = {i, k, j};
        std::array<size_t, 3> jik = {j, i, k};
        std::array<size_t, 3> jki = {j, k, i};
        std::array<size_t, 3> kij = {k, i, j};
        std::array<size_t, 3> kji = {k, j, i};

        // for RHS ={a, b, c}
        CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<3>(
                  index_order_abc, index_order_abc, {{i, j, k}}) == ijk);
        CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<3>(
                  index_order_acb, index_order_abc, {{i, j, k}}) == ikj);
        CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<3>(
                  index_order_bac, index_order_abc, {{i, j, k}}) == jik);
        CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<3>(
                  index_order_bca, index_order_abc, {{i, j, k}}) == kij);
        CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<3>(
                  index_order_cab, index_order_abc, {{i, j, k}}) == jki);
        CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<3>(
                  index_order_cba, index_order_abc, {{i, j, k}}) == kji);
      }
    }
  }
}

// this should iterate over all possible non-symmetric symmetry combinations
template <typename Datatype, typename IndexList, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          size_t SpatialDimA, size_t SpatialDimB, size_t SpatialDimC>
void test_compute_rhs_tensor_index_rank_3_core_no_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c) {
  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<1, 2, 3>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);

  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<1, 3, 2>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);

  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<2, 1, 3>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);

  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<2, 3, 1>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);

  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<3, 1, 2>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);

  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<3, 2, 1>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);
}

// this should iterate over combinations where first two indices are symmetric
template <typename Datatype, typename IndexList, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          size_t SpatialDimA, size_t SpatialDimB, size_t SpatialDimC>
void test_compute_rhs_tensor_index_rank_3_core_ab_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c) {
  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<1, 1, 2>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);

  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<2, 2, 1>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);
}

// this should iterate over combinations where first and third
// indices are symmetric
template <typename Datatype, typename IndexList, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          size_t SpatialDimA, size_t SpatialDimB, size_t SpatialDimC>
void test_compute_rhs_tensor_index_rank_3_core_ac_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c) {
  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<1, 2, 1>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);

  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<2, 1, 2>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);
}

// this should iterate over combinations where second and third
// indices are symmetric
template <typename Datatype, typename IndexList, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          size_t SpatialDimA, size_t SpatialDimB, size_t SpatialDimC>
void test_compute_rhs_tensor_index_rank_3_core_bc_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c) {
  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<1, 2, 2>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);

  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<2, 1, 1>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);
}

// when all three indices are symmetric
template <typename Datatype, typename IndexList, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          size_t SpatialDimA, size_t SpatialDimB, size_t SpatialDimC>
void test_compute_rhs_tensor_index_rank_3_core_abc_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c) {
  test_compute_rhs_tensor_index_rank_3_core_impl<
      Datatype, Symmetry<1, 1, 1>, IndexList, TensorIndexTypeA,
      TensorIndexTypeB, TensorIndexTypeC>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c,
      SpatialDimA, SpatialDimB, SpatialDimC);
}

// Call test_compute_rhs_tensor_index_rank_3_core_no_symmetry
// for all dimension combinations with grid and inertial frames
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          typename TensorIndexC,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3_no_symmetry(
    const TensorIndexA& tensor_index_a, const TensorIndexB& tensor_index_b,
    const TensorIndexC& tensor_index_c) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(2, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(3, data)

#define CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_NO_SYMMETRY(_, data) \
  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<                    \
      Datatype,                                                             \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,      \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,      \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>,     \
      TensorIndexA, TensorIndexB, TensorIndexC, DIM_A(data), DIM_B(data),   \
      DIM_C(data)>(tensor_index_a, tensor_index_b, tensor_index_c);

  GENERATE_INSTANTIATIONS(
      CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_NO_SYMMETRY, (1, 2, 3),
      (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_NO_SYMMETRY
#undef FRAME
#undef DIM_C
#undef DIM_B
#undef DIM_A
}

// Call test_compute_rhs_tensor_index_rank_3_core_ab_symmetry
// for all dimension combinations with grid and inertial frames
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          typename TensorIndexC,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3_ab_symmetry(
    const TensorIndexA& tensor_index_a, const TensorIndexB& tensor_index_b,
    const TensorIndexC& tensor_index_c) {
#define DIM_AB(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_AB_SYMMETRY(_, data) \
  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<                    \
      Datatype,                                                             \
      index_list<TensorIndexTypeA<DIM_AB(data), ValenceA, FRAME(data)>,     \
                 TensorIndexTypeB<DIM_AB(data), ValenceB, FRAME(data)>,     \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>,     \
      TensorIndexA, TensorIndexB, TensorIndexC, DIM_AB(data), DIM_AB(data), \
      DIM_C(data)>(tensor_index_a, tensor_index_b, tensor_index_c);

  GENERATE_INSTANTIATIONS(
      CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_AB_SYMMETRY, (1, 2, 3),
      (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_AB_SYMMETRY
#undef FRAME
#undef DIM_C
#undef DIM_AB
}

// Call test_compute_rhs_tensor_index_rank_3_core_ac_symmetry
// for all dimension combinations with grid and inertial frames
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          typename TensorIndexC,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3_ac_symmetry(
    const TensorIndexA& tensor_index_a, const TensorIndexB& tensor_index_b,
    const TensorIndexC& tensor_index_c) {
#define DIM_AC(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_AC_SYMMETRY(_, data) \
  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<                    \
      Datatype,                                                             \
      index_list<TensorIndexTypeA<DIM_AC(data), ValenceA, FRAME(data)>,     \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,      \
                 TensorIndexTypeC<DIM_AC(data), ValenceC, FRAME(data)>>,    \
      TensorIndexA, TensorIndexB, TensorIndexC, DIM_AC(data), DIM_B(data),  \
      DIM_AC(data)>(tensor_index_a, tensor_index_b, tensor_index_c);

  GENERATE_INSTANTIATIONS(
      CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_AC_SYMMETRY, (1, 2, 3),
      (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_AC_SYMMETRY
#undef FRAME
#undef DIM_B
#undef DIM_AC
}

// Call test_compute_rhs_tensor_index_rank_3_core_bc_symmetry
// for all dimension combinations with grid and inertial frames
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          typename TensorIndexC,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3_bc_symmetry(
    const TensorIndexA& tensor_index_a, const TensorIndexB& tensor_index_b,
    const TensorIndexC& tensor_index_c) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_BC(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_BC_SYMMETRY(_, data) \
  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<                    \
      Datatype,                                                             \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,      \
                 TensorIndexTypeB<DIM_BC(data), ValenceB, FRAME(data)>,     \
                 TensorIndexTypeC<DIM_BC(data), ValenceC, FRAME(data)>>,    \
      TensorIndexA, TensorIndexB, TensorIndexC, DIM_A(data), DIM_BC(data),  \
      DIM_BC(data)>(tensor_index_a, tensor_index_b, tensor_index_c);

  GENERATE_INSTANTIATIONS(
      CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_BC_SYMMETRY, (1, 2, 3),
      (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_BC_SYMMETRY
#undef FRAME
#undef DIM_BC
#undef DIM_A
}

// Call test_compute_rhs_tensor_index_rank_3_core_abc_symmetry
// for all dimension combinations with grid and inertial frames
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexA, typename TensorIndexB,
          typename TensorIndexC,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3_abc_symmetry(
    const TensorIndexA& tensor_index_a, const TensorIndexB& tensor_index_b,
    const TensorIndexC& tensor_index_c) {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_ABC_SYMMETRY(_, data) \
  test_compute_rhs_tensor_index_rank_3_core_abc_symmetry<                    \
      Datatype,                                                              \
      index_list<TensorIndexTypeA<DIM(data), ValenceA, FRAME(data)>,         \
                 TensorIndexTypeB<DIM(data), ValenceB, FRAME(data)>,         \
                 TensorIndexTypeC<DIM(data), ValenceC, FRAME(data)>>,        \
      TensorIndexA, TensorIndexB, TensorIndexC, DIM(data), DIM(data),        \
      DIM(data)>(tensor_index_a, tensor_index_b, tensor_index_c);

  GENERATE_INSTANTIATIONS(
      CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_ABC_SYMMETRY, (1, 2, 3),
      (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TNESOR_INDEX_RANK_3_CORE_ABC_SYMMETRY
#undef FRAME
#undef DIM
}
