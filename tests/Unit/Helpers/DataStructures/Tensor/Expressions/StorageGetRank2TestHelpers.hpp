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
#include "Utilities/TMPL.hpp"

// figure out how to note need to pass in spatial dims and tensor index types
template <typename Datatype, typename Symmetry, typename IndexList,
          typename TensorIndexTypeA, typename TensorIndexTypeB>
void test_storage_get_rank_2_core_impl(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b, const size_t& spatial_dim_1,
    const size_t& spatial_dim_2) {
  Tensor<Datatype, Symmetry, IndexList> rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);

  /*std::cout << "\n\n Rank 2 tensor" << std::endl
            << "spatial_dim_1 : " << spatial_dim_1 << std::endl
            << "spatial_dim_2 : " << spatial_dim_2 << std::endl;*/

  // TODO : need to specialize the below into multiple functions because
  // not all of these expressions can be evaluated - all of these
  // only work if the indices are symmetric
  auto ab_to_ab =
      TensorExpressions::evaluate<TensorIndexTypeA, TensorIndexTypeB>(
          rhs_tensor(tensor_index_type_a, tensor_index_type_b));  // i j -> i j

  /*auto ba_to_ab = TensorExpressions::evaluate<
                        TensorIndexTypeA, TensorIndexTypeB>(
      rhs_tensor(tensor_index_type_b, tensor_index_type_a));  // i j -> j i*/

  // Not compiling either due to an outside error with collapsed_to_storage
  // (or something related to it) or my own compute_map or
  // (storage) get function
  auto ab_to_ba =
      TensorExpressions::evaluate<TensorIndexTypeB, TensorIndexTypeA>(
          rhs_tensor(tensor_index_type_a, tensor_index_type_b));  // i j -> j i

  /*auto ba_to_ba = TensorExpressions::evaluate<
                        TensorIndexTypeB, TensorIndexTypeA>(
      rhs_tensor(tensor_index_type_b, tensor_index_type_a));  // i j -> i j*/

  for (size_t i = 0; i < spatial_dim_1; ++i) {
    for (size_t j = 0; j < spatial_dim_2; ++j) {
      // OLD - don't need all of theseS
      /*CHECK(ab_to_ab.get(i, j) == rhs_tensor.get(i, j));
      CHECK(ba_to_ab.get(j, i) == rhs_tensor.get(i, j));
      CHECK(ab_to_ba.get(j, i) == rhs_tensor.get(i, j));
      CHECK(ba_to_ba.get(i, j) == rhs_tensor.get(i, j));

      CHECK(ab_to_ab.get(i, j) == ba_to_ab.get(j, i));
      CHECK(ab_to_ab.get(i, j) == ab_to_ba.get(j, i));
      CHECK(ab_to_ab.get(i, j) == ba_to_ba.get(i, j));
      CHECK(ba_to_ab.get(j, i) == ab_to_ba.get(j, i));
      CHECK(ba_to_ab.get(j, i) == ba_to_ba.get(i, j));
      CHECK(ab_to_ba.get(j, i) == ba_to_ba.get(i, j));*/

      // uncomment the below once the ab_to_ba is fixed above
      CHECK(ab_to_ab.get(i, j) == rhs_tensor.get(i, j));
      CHECK(ab_to_ba.get(j, i) == rhs_tensor.get(i, j));
      CHECK(ab_to_ab.get(i, j) == ab_to_ba.get(j, i));
    }
  }
}

// this should iterate over all possible non-symmetric symmetry combinations
template <typename Datatype, typename IndexList, typename TensorIndexTypeA,
          typename TensorIndexTypeB, size_t SpatialDim1, size_t SpatialDim2>
void test_storage_get_rank_2_core_no_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b) {
  test_storage_get_rank_2_core_impl<Datatype, Symmetry<1, 2>, IndexList,
                                    TensorIndexTypeA, TensorIndexTypeB>(
      tensor_index_type_a, tensor_index_type_b, SpatialDim1, SpatialDim2);
  test_storage_get_rank_2_core_impl<Datatype, Symmetry<2, 1>, IndexList,
                                    TensorIndexTypeA, TensorIndexTypeB>(
      tensor_index_type_a, tensor_index_type_b, SpatialDim1, SpatialDim2);
}

// this should iterate over all possible non-symmetric symmetry combinations
template <typename Datatype, typename IndexList, typename TensorIndexTypeA,
          typename TensorIndexTypeB, size_t SpatialDim1, size_t SpatialDim2>
void test_storage_get_rank_2_core_symmetric(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b) {
  test_storage_get_rank_2_core_impl<Datatype, Symmetry<1, 1>, IndexList,
                                    TensorIndexTypeA, TensorIndexTypeB>(
      tensor_index_type_a, tensor_index_type_b, SpatialDim1, SpatialDim2);
}

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexTypeA,
          typename TensorIndexTypeB,
          template <size_t, UpLo, typename> class IndexTypeA,
          template <size_t, UpLo, typename> class IndexTypeB, UpLo ValenceA,
          UpLo ValenceB>
void test_storage_get_rank_2_no_symmetry(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b) {
  // Testing all dimension combinations with grid frame
  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  // Testing all dimension combinations with inertial frame
  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_no_symmetry<
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
void test_storage_get_rank_2_symmetric(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b) {
  // Testing all dimension combinations with grid frame
  test_storage_get_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  // Testing all dimension combinations with inertial frame
  test_storage_get_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_storage_get_rank_2_core_symmetric<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 3>(tensor_index_type_a,
                                                tensor_index_type_b);
}
