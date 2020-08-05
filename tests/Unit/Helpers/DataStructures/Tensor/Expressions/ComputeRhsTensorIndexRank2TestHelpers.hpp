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

// instead, this should iterate over
// 0 - SpatialDim1 (all i) and 0 - SpatialDim2 (all j)
// which means moving the for loops out of the test_rank_2
template <typename Datatype, typename Symmetry, typename IndexList,
          typename TensorIndexTypeA, typename TensorIndexTypeB>
void test_compute_rhs_tensor_index_rank_2_core_impl(
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

// this should iterate over all possible non-symmetric symmetry combinations
template <typename Datatype, typename IndexList, typename TensorIndexTypeA,
          typename TensorIndexTypeB, size_t SpatialDimA, size_t SpatialDimB>
void test_compute_rhs_tensor_index_rank_2_core_no_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b) {
  test_compute_rhs_tensor_index_rank_2_core_impl<
      Datatype, Symmetry<1, 2>, IndexList, TensorIndexTypeA, TensorIndexTypeB>(
      tensor_index_type_a, tensor_index_type_b, SpatialDimA, SpatialDimB);
  test_compute_rhs_tensor_index_rank_2_core_impl<
      Datatype, Symmetry<2, 1>, IndexList, TensorIndexTypeA, TensorIndexTypeB>(
      tensor_index_type_a, tensor_index_type_b, SpatialDimA, SpatialDimB);
}

// this should iterate over all possible non-symmetric symmetry combinations
template <typename Datatype, typename IndexList, typename TensorIndexTypeA,
          typename TensorIndexTypeB, size_t SpatialDimA, size_t SpatialDimB>
void test_compute_rhs_tensor_index_rank_2_core_symmetric(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b) {
  test_compute_rhs_tensor_index_rank_2_core_impl<
      Datatype, Symmetry<1, 1>, IndexList, TensorIndexTypeA, TensorIndexTypeB>(
      tensor_index_type_a, tensor_index_type_b, SpatialDimA, SpatialDimB);
}

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexTypeA,
          typename TensorIndexTypeB,
          template <size_t, UpLo, typename> class IndexTypeA,
          template <size_t, UpLo, typename> class IndexTypeB, UpLo ValenceA,
          UpLo ValenceB>
void test_compute_rhs_tensor_index_rank_2_no_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b) {
  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 3>(tensor_index_type_a,
                                                tensor_index_type_b);
}

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexTypeA,
          typename TensorIndexTypeB,
          template <size_t, UpLo, typename> class IndexTypeA,
          template <size_t, UpLo, typename> class IndexTypeB, UpLo ValenceA,
          UpLo ValenceB>
void test_compute_rhs_tensor_index_rank_2_symmetric(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b) {
  test_compute_rhs_tensor_index_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 3>(tensor_index_type_a,
                                                tensor_index_type_b);
}
