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

template <typename Datatype, typename Symmetry, typename IndexList,
          typename TensorIndexTypeA, typename TensorIndexTypeB>
void test_compute_rhs_tensor_index_rank_2_helper(
    const TensorIndexTypeA& index_type_a, const TensorIndexTypeB& index_type_b,
    const size_t& i, const size_t& j) {
  Tensor<Datatype, Symmetry, IndexList> rhs_tensor{};

  auto rhs_tensor_ab = rhs_tensor(index_type_a, index_type_b);
  auto rhs_tensor_ba = rhs_tensor(index_type_b, index_type_a);

  std::array<size_t, 2> index_order_ab = {TensorIndexTypeA::value,
                                          TensorIndexTypeB::value};
  std::array<size_t, 2> index_order_ba = {TensorIndexTypeB::value,
                                          TensorIndexTypeA::value};

  /*std::array<std::array<size_t, 2>, 2> index_orderings_rank2 =
       {index_order_ab, index_order_ba};*/

  for (int n = 0; n < 2; n++) {
    /*size_t i = index_orderings_rank2[n][0];
    size_t j = index_orderings_rank2[n][1];*/

    std::array<size_t, 2> ij = {i, j};
    std::array<size_t, 2> ji = {j, i};

    CHECK(rhs_tensor_ab.template compute_rhs_tensor_index<2>(
              index_order_ab, index_order_ab, {{i, j}}) == ij);
    CHECK(rhs_tensor_ab.template compute_rhs_tensor_index<2>(
              index_order_ba, index_order_ab, {{i, j}}) == ji);
    CHECK(rhs_tensor_ba.template compute_rhs_tensor_index<2>(
              index_order_ba, index_order_ba, {{i, j}}) == ij);
    CHECK(rhs_tensor_ba.template compute_rhs_tensor_index<2>(
              index_order_ab, index_order_ba, {{i, j}}) == ji);
  }
}

// TODO - make this templated on Symm, Indices...
//      - i, j, and k are not supposed to be the same as generic indices,
//        they can be whatever value.
//      - Should take arguments:
//        - lhs_tensor_index (i, j, k) - either array or 3 size_t args
template <typename Datatype, typename Symmetry, typename IndexList,
          typename TensorIndexTypeA, typename TensorIndexTypeB,
          typename TensorIndexTypeC>
void test_compute_rhs_tensor_index_rank_3_helper(
    const TensorIndexTypeA& index_type_a, const TensorIndexTypeB& index_type_b,
    const TensorIndexTypeC& index_type_c, const size_t& i, const size_t& j,
    const size_t& k) {

  Tensor<Datatype, Symmetry, IndexList> rhs_tensor{};

  auto rhs_tensor_abc = rhs_tensor(index_type_a, index_type_b, index_type_c);
  auto rhs_tensor_acb = rhs_tensor(index_type_a, index_type_c, index_type_b);
  auto rhs_tensor_bac = rhs_tensor(index_type_b, index_type_a, index_type_c);
  auto rhs_tensor_bca = rhs_tensor(index_type_b, index_type_c, index_type_a);
  auto rhs_tensor_cab = rhs_tensor(index_type_c, index_type_a, index_type_b);
  auto rhs_tensor_cba = rhs_tensor(index_type_c, index_type_b, index_type_a);

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

  for (int n = 0; n < 6; n++) {
    std::array<size_t, 3> ijk = {i, j, k};
    std::array<size_t, 3> ikj = {i, k, j};
    std::array<size_t, 3> jik = {j, i, k};
    std::array<size_t, 3> jki = {j, k, i};
    std::array<size_t, 3> kij = {k, i, j};
    std::array<size_t, 3> kji = {k, j, i};

    // auto result = rhs_tensor_abc.template
    // compute_rhs_tensor_index<3>(index_order_abc, index_order_abc, ijk); bool
    // flag = (result == ijk);

    // for RHS ={a, b, c}
    CHECK(rhs_tensor_abc.template compute_rhs_tensor_index<3>(
              index_order_abc, index_order_abc, {{i, j, k}}) == ijk);
    CHECK(rhs_tensor_abc.template compute_rhs_tensor_index<3>(
              index_order_acb, index_order_abc, {{i, j, k}}) == ikj);
    CHECK(rhs_tensor_abc.template compute_rhs_tensor_index<3>(
              index_order_bac, index_order_abc, {{i, j, k}}) == jik);
    CHECK(rhs_tensor_abc.template compute_rhs_tensor_index<3>(
              index_order_bca, index_order_abc, {{i, j, k}}) == kij);
    CHECK(rhs_tensor_abc.template compute_rhs_tensor_index<3>(
              index_order_cab, index_order_abc, {{i, j, k}}) == jki);
    CHECK(rhs_tensor_abc.template compute_rhs_tensor_index<3>(
              index_order_cba, index_order_abc, {{i, j, k}}) == kji);

    // for RHS = {a, c, b}
    CHECK(rhs_tensor_acb.template compute_rhs_tensor_index<3>(
              index_order_abc, index_order_acb, {{i, j, k}}) == ikj);
    CHECK(rhs_tensor_acb.template compute_rhs_tensor_index<3>(
              index_order_acb, index_order_acb, {{i, j, k}}) == ijk);
    CHECK(rhs_tensor_acb.template compute_rhs_tensor_index<3>(
              index_order_bac, index_order_acb, {{i, j, k}}) == jki);
    CHECK(rhs_tensor_acb.template compute_rhs_tensor_index<3>(
              index_order_bca, index_order_acb, {{i, j, k}}) == kji);
    CHECK(rhs_tensor_acb.template compute_rhs_tensor_index<3>(
              index_order_cab, index_order_acb, {{i, j, k}}) == jik);
    CHECK(rhs_tensor_acb.template compute_rhs_tensor_index<3>(
              index_order_cba, index_order_acb, {{i, j, k}}) == kij);

    // for RHS = {b, a, c}
    CHECK(rhs_tensor_bac.template compute_rhs_tensor_index<3>(
              index_order_abc, index_order_bac, {{i, j, k}}) == jik);
    CHECK(rhs_tensor_bac.template compute_rhs_tensor_index<3>(
              index_order_acb, index_order_bac, {{i, j, k}}) == kij);
    CHECK(rhs_tensor_bac.template compute_rhs_tensor_index<3>(
              index_order_bac, index_order_bac, {{i, j, k}}) == ijk);
    CHECK(rhs_tensor_bac.template compute_rhs_tensor_index<3>(
              index_order_bca, index_order_bac, {{i, j, k}}) == ikj);
    CHECK(rhs_tensor_bac.template compute_rhs_tensor_index<3>(
              index_order_cab, index_order_bac, {{i, j, k}}) == kji);
    CHECK(rhs_tensor_bac.template compute_rhs_tensor_index<3>(
              index_order_cba, index_order_bac, {{i, j, k}}) == jki);

    // for RHS = {b, c, a}
    CHECK(rhs_tensor_bca.template compute_rhs_tensor_index<3>(
              index_order_abc, index_order_bca, {{i, j, k}}) == jki);
    CHECK(rhs_tensor_bca.template compute_rhs_tensor_index<3>(
              index_order_acb, index_order_bca, {{i, j, k}}) == kji);
    CHECK(rhs_tensor_bca.template compute_rhs_tensor_index<3>(
              index_order_bac, index_order_bca, {{i, j, k}}) == ikj);
    CHECK(rhs_tensor_bca.template compute_rhs_tensor_index<3>(
              index_order_bca, index_order_bca, {{i, j, k}}) == ijk);
    CHECK(rhs_tensor_bca.template compute_rhs_tensor_index<3>(
              index_order_cab, index_order_bca, {{i, j, k}}) == kij);
    CHECK(rhs_tensor_bca.template compute_rhs_tensor_index<3>(
              index_order_cba, index_order_bca, {{i, j, k}}) == jik);

    // for RHS = {c, a, b}
    CHECK(rhs_tensor_cab.template compute_rhs_tensor_index<3>(
              index_order_abc, index_order_cab, {{i, j, k}}) == kij);
    CHECK(rhs_tensor_cab.template compute_rhs_tensor_index<3>(
              index_order_acb, index_order_cab, {{i, j, k}}) == jik);
    CHECK(rhs_tensor_cab.template compute_rhs_tensor_index<3>(
              index_order_bac, index_order_cab, {{i, j, k}}) == kji);
    CHECK(rhs_tensor_cab.template compute_rhs_tensor_index<3>(
              index_order_bca, index_order_cab, {{i, j, k}}) == jki);
    CHECK(rhs_tensor_cab.template compute_rhs_tensor_index<3>(
              index_order_cab, index_order_cab, {{i, j, k}}) == ijk);
    CHECK(rhs_tensor_cab.template compute_rhs_tensor_index<3>(
              index_order_cba, index_order_cab, {{i, j, k}}) == ikj);

    // for RHS = {c, b, a}
    CHECK(rhs_tensor_cba.template compute_rhs_tensor_index<3>(
              index_order_abc, index_order_cba, {{i, j, k}}) == kji);
    CHECK(rhs_tensor_cba.template compute_rhs_tensor_index<3>(
              index_order_acb, index_order_cba, {{i, j, k}}) == jki);
    CHECK(rhs_tensor_cba.template compute_rhs_tensor_index<3>(
              index_order_bac, index_order_cba, {{i, j, k}}) == kij);
    CHECK(rhs_tensor_cba.template compute_rhs_tensor_index<3>(
              index_order_bca, index_order_cba, {{i, j, k}}) == jik);
    CHECK(rhs_tensor_cba.template compute_rhs_tensor_index<3>(
              index_order_cab, index_order_cba, {{i, j, k}}) == ikj);
    CHECK(rhs_tensor_cba.template compute_rhs_tensor_index<3>(
              index_order_cba, index_order_cba, {{i, j, k}}) == ijk);
  }
}


SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.ComputeRhsTensorIndex",
                  "[DataStructures][Unit]") {
  const size_t num_indices_rank_2 = 3;
  using IndexListRank2 =
      index_list<SpatialIndex<num_indices_rank_2, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<num_indices_rank_2, UpLo::Lo, Frame::Grid>>;

  for (size_t i = 0; i < num_indices_rank_2; i++) {
    for (size_t j = 0; j < num_indices_rank_2; j++) {
      test_compute_rhs_tensor_index_rank_2_helper<
          double, Symmetry<1, 1>, IndexListRank2, ti_a_t, ti_b_t>(ti_a, ti_b, i,
                                                                  j);
      test_compute_rhs_tensor_index_rank_2_helper<
          double, Symmetry<1, 2>, IndexListRank2, ti_a_t, ti_b_t>(ti_a, ti_b, i,
                                                                  j);
      test_compute_rhs_tensor_index_rank_2_helper<
          double, Symmetry<1, 1>, IndexListRank2, ti_g_t, ti_c_t>(ti_g, ti_c, i,
                                                                  j);
      test_compute_rhs_tensor_index_rank_2_helper<
          double, Symmetry<2, 1>, IndexListRank2, ti_g_t, ti_c_t>(ti_g, ti_c, i,
                                                                  j);
      test_compute_rhs_tensor_index_rank_2_helper<
          double, Symmetry<1, 1>, IndexListRank2, ti_C_t, ti_d_t>(ti_C, ti_d, i,
                                                                  j);
      test_compute_rhs_tensor_index_rank_2_helper<
          double, Symmetry<1, 2>, IndexListRank2, ti_C_t, ti_d_t>(ti_C, ti_d, i,
                                                                  j);
    }
  }

  const size_t num_indices_rank_3 = 5;
  using IndexListRank3 =
      index_list<SpatialIndex<num_indices_rank_3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<num_indices_rank_3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<num_indices_rank_3, UpLo::Lo, Frame::Grid>>;

  for (size_t i = 0; i < num_indices_rank_3; i++) {
    for (size_t j = 0; j < num_indices_rank_3; j++) {
      for (size_t k = 0; k < num_indices_rank_3; k++) {
        test_compute_rhs_tensor_index_rank_3_helper<
            double, Symmetry<1, 1, 1>, IndexListRank3, ti_a_t, ti_b_t, ti_c_t>(
            ti_a, ti_b, ti_c, i, j, k);
        test_compute_rhs_tensor_index_rank_3_helper<
            double, Symmetry<1, 2, 1>, IndexListRank3, ti_a_t, ti_b_t, ti_c_t>(
            ti_a, ti_b, ti_c, i, j, k);
        test_compute_rhs_tensor_index_rank_3_helper<
            double, Symmetry<2, 3, 1>, IndexListRank3, ti_a_t, ti_b_t, ti_c_t>(
            ti_a, ti_b, ti_c, i, j, k);

        test_compute_rhs_tensor_index_rank_3_helper<
            double, Symmetry<1, 1, 1>, IndexListRank3, ti_c_t, ti_b_t, ti_a_t>(
            ti_c, ti_b, ti_a, i, j, k);
        test_compute_rhs_tensor_index_rank_3_helper<
            double, Symmetry<2, 1, 2>, IndexListRank3, ti_c_t, ti_b_t, ti_a_t>(
            ti_c, ti_b, ti_a, i, j, k);
        test_compute_rhs_tensor_index_rank_3_helper<
            double, Symmetry<3, 1, 2>, IndexListRank3, ti_c_t, ti_b_t, ti_a_t>(
            ti_c, ti_b, ti_a, i, j, k);

        test_compute_rhs_tensor_index_rank_3_helper<
            double, Symmetry<1, 1, 1>, IndexListRank3, ti_b_t, ti_D_t, ti_A_t>(
            ti_b, ti_D, ti_A, i, j, k);
        test_compute_rhs_tensor_index_rank_3_helper<
            double, Symmetry<1, 1, 2>, IndexListRank3, ti_b_t, ti_D_t, ti_A_t>(
            ti_b, ti_D, ti_A, i, j, k);
        test_compute_rhs_tensor_index_rank_3_helper<
            double, Symmetry<2, 1, 3>, IndexListRank3, ti_b_t, ti_D_t, ti_A_t>(
            ti_b, ti_D, ti_A, i, j, k);
      }
    }
  }
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
