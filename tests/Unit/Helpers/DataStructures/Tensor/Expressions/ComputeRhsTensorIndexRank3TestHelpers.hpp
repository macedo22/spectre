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

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          template <size_t, UpLo, typename> class IndexTypeA,
          template <size_t, UpLo, typename> class IndexTypeB,
          template <size_t, UpLo, typename> class IndexTypeC, UpLo ValenceA,
          UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3_no_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c){
  // Testing all dimension combinations with grid frame
  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  // Testing all dimension combinations with inertial frame
  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);
}

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          template <size_t, UpLo, typename> class IndexTypeA,
          template <size_t, UpLo, typename> class IndexTypeB,
          template <size_t, UpLo, typename> class IndexTypeC, UpLo ValenceA,
          UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3_ab_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c){
  // Testing all dimension combinations with grid frame
  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  // Testing all dimension combinations with inertial frame
    test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ab_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);
}

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          template <size_t, UpLo, typename> class IndexTypeA,
          template <size_t, UpLo, typename> class IndexTypeB,
          template <size_t, UpLo, typename> class IndexTypeC, UpLo ValenceA,
          UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3_ac_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c){
  // Testing all dimension combinations with grid frame
  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  // Testing all dimension combinations with inertial frame
  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_ac_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);
}

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          template <size_t, UpLo, typename> class IndexTypeA,
          template <size_t, UpLo, typename> class IndexTypeB,
          template <size_t, UpLo, typename> class IndexTypeC, UpLo ValenceA,
          UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3_bc_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c){
  // Testing all dimension combinations with grid frame
  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  // Testing all dimension combinations with inertial frame
    test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_bc_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);
}

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          template <size_t, UpLo, typename> class IndexTypeA,
          template <size_t, UpLo, typename> class IndexTypeB,
          template <size_t, UpLo, typename> class IndexTypeC, UpLo ValenceA,
          UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3_abc_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c){
  // Testing all dimension combinations with grid frame
  test_compute_rhs_tensor_index_rank_3_core_abc_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_abc_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_abc_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

    // Testing all dimension combinations with ginertial frame
  test_compute_rhs_tensor_index_rank_3_core_abc_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_abc_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core_abc_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);
}
