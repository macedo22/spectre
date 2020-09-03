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
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_core(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) {
  const Tensor<Datatype, RhsSymmetry, RhsTensorIndexTypeList> rhs_tensor{};
  const auto R_abc = rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c);

  const size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<RhsTensorIndexTypeList, 2>::dim;

  const std::array<size_t, 3> index_order_abc = {
      TensorIndexA::value, TensorIndexB::value, TensorIndexC::value};
  const std::array<size_t, 3> index_order_acb = {
      TensorIndexA::value, TensorIndexC::value, TensorIndexB::value};
  const std::array<size_t, 3> index_order_bac = {
      TensorIndexB::value, TensorIndexA::value, TensorIndexC::value};
  const std::array<size_t, 3> index_order_bca = {
      TensorIndexB::value, TensorIndexC::value, TensorIndexA::value};
  const std::array<size_t, 3> index_order_cab = {
      TensorIndexC::value, TensorIndexA::value, TensorIndexB::value};
  const std::array<size_t, 3> index_order_cba = {
      TensorIndexC::value, TensorIndexB::value, TensorIndexA::value};

  for (size_t i = 0; i < dim_a; i++) {
    for (size_t j = 0; j < dim_b; j++) {
      for (size_t k = 0; k < dim_c; k++) {
        const std::array<size_t, 3> ijk = {i, j, k};
        const std::array<size_t, 3> ikj = {i, k, j};
        const std::array<size_t, 3> jik = {j, i, k};
        const std::array<size_t, 3> jki = {j, k, i};
        const std::array<size_t, 3> kij = {k, i, j};
        const std::array<size_t, 3> kji = {k, j, i};

        // RHS = {a, b, c}
        CHECK(R_abc.template compute_rhs_tensor_index<3>(
                  index_order_abc, index_order_abc, ijk) == ijk);
        CHECK(R_abc.template compute_rhs_tensor_index<3>(
                  index_order_acb, index_order_abc, ijk) == ikj);
        CHECK(R_abc.template compute_rhs_tensor_index<3>(
                  index_order_bac, index_order_abc, ijk) == jik);
        CHECK(R_abc.template compute_rhs_tensor_index<3>(
                  index_order_bca, index_order_abc, ijk) == kij);
        CHECK(R_abc.template compute_rhs_tensor_index<3>(
                  index_order_cab, index_order_abc, ijk) == jki);
        CHECK(R_abc.template compute_rhs_tensor_index<3>(
                  index_order_cba, index_order_abc, ijk) == kji);
      }
    }
  }
}

// Test all dimension combinations with grid and inertial frames
// for nonsymmetric indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_no_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(2, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(3, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE(_, data)          \
  test_compute_rhs_tensor_index_rank_3_core<                             \
      Datatype, Symmetry<3, 2, 1>,                                       \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,   \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,   \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE,
                          (1, 2, 3), (1, 2, 3), (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE
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
template <typename Datatype,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_ab_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) {
#define DIM_AB(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE(_, data)          \
  test_compute_rhs_tensor_index_rank_3_core<                             \
      Datatype, Symmetry<2, 2, 1>,                                       \
      index_list<TensorIndexTypeA<DIM_AB(data), ValenceA, FRAME(data)>,  \
                 TensorIndexTypeB<DIM_AB(data), ValenceB, FRAME(data)>,  \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE,
                          (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE
#undef FRAME
#undef DIM_C
#undef DIM_AB
}

// Test all dimension combinations with grid and inertial frames
// for symmetric first and third indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_ac_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) {
#define DIM_AC(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE(_, data)           \
  test_compute_rhs_tensor_index_rank_3_core<                              \
      Datatype, Symmetry<2, 1, 2>,                                        \
      index_list<TensorIndexTypeA<DIM_AC(data), ValenceA, FRAME(data)>,   \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,    \
                 TensorIndexTypeC<DIM_AC(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE,
                          (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE
#undef FRAME
#undef DIM_B
#undef DIM_AC
}

// Test all dimension combinations with grid and inertial frames
// for symmetric second and third indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_bc_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_BC(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE(_, data)           \
  test_compute_rhs_tensor_index_rank_3_core<                              \
      Datatype, Symmetry<2, 1, 1>,                                        \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,    \
                 TensorIndexTypeB<DIM_BC(data), ValenceB, FRAME(data)>,   \
                 TensorIndexTypeC<DIM_BC(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE,
                          (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE
#undef FRAME
#undef DIM_BC
#undef DIM_A
}

// Test all dimensions with grid and inertial frames
// for symmetric indices
//
// TensorIndex refers to TensorIndex<#>
// TensorIndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_abc_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE(_, data)        \
  test_compute_rhs_tensor_index_rank_3_core<                           \
      Datatype, Symmetry<1, 1, 1>,                                     \
      index_list<TensorIndexTypeA<DIM(data), ValenceA, FRAME(data)>,   \
                 TensorIndexTypeB<DIM(data), ValenceB, FRAME(data)>,   \
                 TensorIndexTypeC<DIM(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE,
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_CORE
#undef FRAME
#undef DIM
}
