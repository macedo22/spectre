// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

// include on multiple lines just to avoid line length warning
//#include
//"tests/Unit/DataStructures/Tensor/Expressions/
// ComputeRhsTensorIndex_Test_Helpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.TryToDoTensorStuff",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);

  // try to get the TensorIndexTypeA::value and TensorIndexTypeB::value
  // can also try to get the dimension of both indices, but I can
  // also just pass those in...
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.StorageGet",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);

  auto result1 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_a, ti_b));

  auto result2 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_b, ti_a));

  auto result3 = TensorExpressions::evaluate<ti_b_t, ti_a_t>(All(ti_a, ti_b));

  auto result4 = TensorExpressions::evaluate<ti_b_t, ti_a_t>(All(ti_b, ti_a));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      CHECK(result1.get(i, j) == All.get(i, j));
      CHECK(result2.get(j, i) == All.get(i, j));
      CHECK(result3.get(j, i) == All.get(i, j));
      CHECK(result4.get(i, j) == All.get(i, j));

      CHECK(result1.get(j, i) == result2.get(i, j));
      CHECK(result1.get(j, i) == result2.get(i, j));
      CHECK(result1.get(i, j) == result4.get(i, j));
      CHECK(result2.get(i, j) == result3.get(i, j));
      CHECK(result2.get(j, i) == result4.get(i, j));
      CHECK(result3.get(j, i) == result4.get(i, j));
    }
  }

  Tensor<double, Symmetry<2, 1, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Auul{};

  auto result5 = TensorExpressions::evaluate<ti_G_t, ti_B_t, ti_d_t>(
      Auul(ti_G, ti_B, ti_d));
  auto result6 = TensorExpressions::evaluate<ti_G_t, ti_B_t, ti_d_t>(
      Auul(ti_d, ti_G, ti_B));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; k++) {
        CHECK(result5.get(i, j, k) == Auul.get(i, j, k));
        CHECK(result6.get(j, k, i) == Auul.get(i, j, k));
        CHECK(result5.get(i, j, k) == result6.get(j, k, i));
      }
    }
  }
}

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
void test_compute_rhs_tensor_index_rank_2_core(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b) {
  /*test_compute_rhs_tensor_index_rank_2_core_impl<
          Datatype, Symmetry<1, 1>, IndexList,
              TensorIndexTypeA, TensorIndexTypeB>(
            tensor_index_type_a, tensor_index_type_b, SpatialDimA,
     SpatialDimB);*/
  test_compute_rhs_tensor_index_rank_2_core_impl<
      Datatype, Symmetry<1, 2>, IndexList, TensorIndexTypeA, TensorIndexTypeB>(
      tensor_index_type_a, tensor_index_type_b, SpatialDimA, SpatialDimB);
  test_compute_rhs_tensor_index_rank_2_core_impl<
      Datatype, Symmetry<2, 1>, IndexList, TensorIndexTypeA, TensorIndexTypeB>(
      tensor_index_type_a, tensor_index_type_b, SpatialDimA, SpatialDimB);
}

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexTypeA,
          typename TensorIndexTypeB,
          template <size_t, UpLo, typename> class IndexTypeA,
          template <size_t, UpLo, typename> class IndexTypeB, UpLo ValenceA,
          UpLo ValenceB>
void test_compute_rhs_tensor_index_rank_2(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b) {
  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  // TODO: it seems to be breaking on this one for some reason?
  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 1, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 2, 3>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 1>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 2>(tensor_index_type_a,
                                                tensor_index_type_b);

  test_compute_rhs_tensor_index_rank_2_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, 3, 3>(tensor_index_type_a,
                                                tensor_index_type_b);
}

// *** The last (outermost) one should go through all
// grid types and spatial dimensions

// TODO - make this templated on Symm, Indices...
//      - i, j, and k are not supposed to be the same as generic indices,
//        they can be whatever value.
//      - Should take arguments:
//        - lhs_tensor_index (i, j, k) - either array or 3 size_t args
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

  /*std::array<std::array<size_t, 3>, 6> index_orderings_rank3 = {
      index_order_abc, index_order_acb, index_order_bac,
      index_order_bca, index_order_cab, index_order_cba};*/

  // auto result = rhs_tensor_abc.template
  // compute_rhs_tensor_index<3>(index_order_abc, index_order_abc, ijk); bool
  // flag = (result == ijk);

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
void test_compute_rhs_tensor_index_rank_3_core(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c) {
  /*test_compute_rhs_tensor_index_rank_2_core_impl<
          Datatype, Symmetry<1, 1>, IndexList,
              TensorIndexTypeA, TensorIndexTypeB>(
            tensor_index_type_a, tensor_index_type_b, SpatialDimA,
     SpatialDimB);*/
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

// TensorIndexType refers to TensorIndex<#>
// IndexType refers to SpatialIndex or SpacetimeIndex
template <typename Datatype, typename TensorIndexTypeA,
          typename TensorIndexTypeB, typename TensorIndexTypeC,
          template <size_t, UpLo, typename> class IndexTypeA,
          template <size_t, UpLo, typename> class IndexTypeB,
          template <size_t, UpLo, typename> class IndexTypeC, UpLo ValenceA,
          UpLo ValenceB, UpLo ValenceC>
void test_compute_rhs_tensor_index_rank_3(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b,
    const TensorIndexTypeC& tensor_index_type_c) {
  // all dimensionality combinations
  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<1, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<2, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<1, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<2, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Grid>,
                 IndexTypeB<3, ValenceB, Frame::Grid>,
                 IndexTypeC<3, ValenceC, Frame::Grid>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  // Testing all dimension combinations with inertial frame
  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<1, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 1, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<2, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 2, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<1, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 1, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<2, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 2, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<1, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 1>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<2, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 2>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);

  test_compute_rhs_tensor_index_rank_3_core<
      Datatype,
      index_list<IndexTypeA<3, ValenceA, Frame::Inertial>,
                 IndexTypeB<3, ValenceB, Frame::Inertial>,
                 IndexTypeC<3, ValenceC, Frame::Inertial>>,
      TensorIndexTypeA, TensorIndexTypeB, TensorIndexTypeC, 3, 3, 3>(
      tensor_index_type_a, tensor_index_type_b, tensor_index_type_c);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.ComputeRhsTensorIndex",
                  "[DataStructures][Unit]") {
  test_compute_rhs_tensor_index_rank_2<double, ti_c_t, ti_I_t, SpatialIndex,
                                       SpacetimeIndex, UpLo::Lo, UpLo::Up>(
      ti_c, ti_I);

  test_compute_rhs_tensor_index_rank_3<
      double, ti_D_t, ti_j_t, ti_B_t, SpatialIndex, SpacetimeIndex,
      SpatialIndex, UpLo::Up, UpLo::Lo, UpLo::Up>(ti_D, ti_j, ti_B);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubtract",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Hll{}, HHll{};
  std::iota(Hll.begin(), Hll.end(), 0.0);
  std::iota(HHll.rbegin(), HHll.rend(), -Hll.size());
  /// [use_tensor_index]
  auto Gll = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_a, ti_b) +
                                                         Hll(ti_a, ti_b));
  auto Gll2 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_a, ti_b) +
                                                          Hll(ti_b, ti_a));
  auto Gll3 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(
      All(ti_a, ti_b) + Hll(ti_b, ti_a) + All(ti_b, ti_a) - Hll(ti_b, ti_a));
  /// [use_tensor_index]
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      CHECK(Gll.get(i, j) == All.get(i, j) + Hll.get(i, j));
      CHECK(Gll2.get(i, j) == All.get(i, j) + Hll.get(j, i));
      CHECK(Gll3.get(i, j) == 2.0 * All.get(i, j));
    }
  }
  // Test 3 indices add subtract
  Tensor<double, Symmetry<1, 1, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Alll{};
  std::iota(Alll.begin(), Alll.end(), 0.0);
  Tensor<double, Symmetry<1, 2, 3>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Hlll{}, HHlll{};
  std::iota(Hlll.begin(), Hlll.end(), 0.0);
  std::iota(HHlll.rbegin(), HHlll.rend(), -Hlll.size());
  auto Glll = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_a, ti_b, ti_c));
  auto Glll2 = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_b, ti_a, ti_c));
  auto Glll3 = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_b, ti_a, ti_c) + Alll(ti_b, ti_a, ti_c) -
      Hlll(ti_b, ti_a, ti_c));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        CHECK(Glll.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(i, j, k));
        CHECK(Glll2.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(j, i, k));
        CHECK(Glll3.get(i, j, k) == 2.0 * Alll.get(i, j, k));
      }
    }
  }
}
