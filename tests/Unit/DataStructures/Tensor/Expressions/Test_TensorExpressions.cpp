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
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.ComputeRhsTensorIndex",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      All_Rhs{};
  auto All_Rhs_ab = All_Rhs(ti_a, ti_b);  // {0, 1}
  auto All_Rhs_ba = All_Rhs(ti_b, ti_a);  // {1, 0}

  std::array<size_t, 2> index_order_ab = {0, 1};
  std::array<size_t, 2> index_order_ba = {1, 0};

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      CHECK(All_Rhs_ab.compute_rhs_tensor_index<2>(
                index_order_ab, index_order_ab, {{i, j}}) == {{i, j}});
      CHECK(All_Rhs_ab.compute_rhs_tensor_index<2>(
                index_order_ba, index_order_ab, {{i, j}}) == {{j, i}});
      CHECK(All_Rhs_ba.compute_rhs_tensor_index<2>(
                index_order_ba, index_order_ba, {{i, j}}) == {{i, j}});
      CHECK(All_Rhs_ba.compute_rhs_tensor_index<2>(
                index_order_ab, index_order_ba, {{i, j}}) == {{j, i}});
    }
  }

  Tensor<double, Symmetry<1, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Alll_Rhs{};
  auto Alll_Rhs_abc = Alll_Rhs(ti_a, ti_b, ti_c);  // {0, 1, 2}
  auto Alll_Rhs_acb = Alll_Rhs(ti_a, ti_c, ti_b);  // {0, 2, 1}
  auto Alll_Rhs_bac = Alll_Rhs(ti_b, ti_a, ti_c);  // {1, 0, 2}
  auto Alll_Rhs_bca = Alll_Rhs(ti_b, ti_c, ti_a);  // {1, 2, 0}
  auto Alll_Rhs_cab = Alll_Rhs(ti_c, ti_a, ti_b);  // {2, 0, 1}
  auto Alll_Rhs_cba = Alll_Rhs(ti_c, ti_b, ti_a);  // {2, 1, 0}

  std::array<size_t, 3> index_order_abc = {0, 1, 2};
  std::array<size_t, 3> index_order_acb = {0, 2, 1};
  std::array<size_t, 3> index_order_bac = {1, 0, 2};
  std::array<size_t, 3> index_order_bca = {1, 2, 0};
  std::array<size_t, 3> index_order_cab = {2, 0, 1};
  std::array<size_t, 3> index_order_cba = {2, 1, 0};

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 3; k++) {
        // for RHS ={a, b, c}
        CHECK(Alll_Rhs_abc.compute_rhs_tensor_index<3>(
                  index_order_abc, index_order_abc, {{i, j, k}}) ==
              {{i, j, k}});
        CHECK(Alll_Rhs_abc.compute_rhs_tensor_index<3>(
                  index_order_acb, index_order_abc, {{i, j, k}}) ==
              {{i, k, j}});
        CHECK(Alll_Rhs_abc.compute_rhs_tensor_index<3>(
                  index_order_bac, index_order_abc, {{i, j, k}}) ==
              {{j, i, k}});
        CHECK(Alll_Rhs_abc.compute_rhs_tensor_index<3>(
                  index_order_bca, index_order_abc, {{i, j, k}}) ==
              {{j, k, i}});
        CHECK(Alll_Rhs_abc.compute_rhs_tensor_index<3>(
                  index_order_cab, index_order_abc, {{i, j, k}}) ==
              {{k, i, j}});
        CHECK(Alll_Rhs_abc.compute_rhs_tensor_index<3>(
                  index_order_cba, index_order_abc, {{i, j, k}}) ==
              {{k, j, i}});

        // for RHS = {a, c, b}
        CHECK(Alll_Rhs_acb.compute_rhs_tensor_index<3>(
                  index_order_abc, index_order_acb, {{i, j, k}}) ==
              {{i, k, j}});
        CHECK(Alll_Rhs_acb.compute_rhs_tensor_index<3>(
                  index_order_acb, index_order_acb, {{i, j, k}}) ==
              {{i, j, k}});
        CHECK(Alll_Rhs_acb.compute_rhs_tensor_index<3>(
                  index_order_bac, index_order_acb, {{i, j, k}}) ==
              {{j, k, i}});
        CHECK(Alll_Rhs_acb.compute_rhs_tensor_index<3>(
                  index_order_bca, index_order_acb, {{i, j, k}}) ==
              {{k, j, i}});
        CHECK(Alll_Rhs_acb.compute_rhs_tensor_index<3>(
                  index_order_cab, index_order_acb, {{i, j, k}}) ==
              {{j, i, k}});
        CHECK(Alll_Rhs_acb.compute_rhs_tensor_index<3>(
                  index_order_cba, index_order_acb, {{i, j, k}}) ==
              {{k, i, j}});

        // for RHS = {b, a, c}
        CHECK(Alll_Rhs_bac.compute_rhs_tensor_index<3>(
                  index_order_abc, index_order_bac, {{i, j, k}}) ==
              {{j, i, k}});
        CHECK(Alll_Rhs_bac.compute_rhs_tensor_index<3>(
                  index_order_acb, index_order_bac, {{i, j, k}}) ==
              {{k, i, j}});
        CHECK(Alll_Rhs_bac.compute_rhs_tensor_index<3>(
                  index_order_bac, index_order_bac, {{i, j, k}}) ==
              {{i, j, k}});
        CHECK(Alll_Rhs_bac.compute_rhs_tensor_index<3>(
                  index_order_bca, index_order_bac, {{i, j, k}}) ==
              {{i, k, j}});
        CHECK(Alll_Rhs_bac.compute_rhs_tensor_index<3>(
                  index_order_cab, index_order_bac, {{i, j, k}}) ==
              {{k, j, i}});
        CHECK(Alll_Rhs_bac.compute_rhs_tensor_index<3>(
                  index_order_cba, index_order_bac, {{i, j, k}}) ==
              {{j, k, i}});

        // for RHS = {b, c, a}
        CHECK(Alll_Rhs_bca.compute_rhs_tensor_index<3>(
                  index_order_abc, index_order_bca, {{i, j, k}}) ==
              {{j, k, i}});
        CHECK(Alll_Rhs_bca.compute_rhs_tensor_index<3>(
                  index_order_acb, index_order_bca, {{i, j, k}}) ==
              {{k, j, i}});
        CHECK(Alll_Rhs_bca.compute_rhs_tensor_index<3>(
                  index_order_bac, index_order_bca, {{i, j, k}}) ==
              {{i, k, j}});
        CHECK(Alll_Rhs_bca.compute_rhs_tensor_index<3>(
                  index_order_bca, index_order_bca, {{i, j, k}}) ==
              {{i, j, k}});
        CHECK(Alll_Rhs_bca.compute_rhs_tensor_index<3>(
                  index_order_cab, index_order_bca, {{i, j, k}}) ==
              {{k, i, j}});
        CHECK(Alll_Rhs_bca.compute_rhs_tensor_index<3>(
                  index_order_cba, index_order_bca, {{i, j, k}}) ==
              {{j, i, k}});

        // for RHS = {c, a, b}
        CHECK(Alll_Rhs_cab.compute_rhs_tensor_index<3>(
                  index_order_abc, index_order_cab, {{i, j, k}}) ==
              {{k, i, j}});
        CHECK(Alll_Rhs_cab.compute_rhs_tensor_index<3>(
                  index_order_acb, index_order_cab, {{i, j, k}}) ==
              {{j, i, k}});
        CHECK(Alll_Rhs_cab.compute_rhs_tensor_index<3>(
                  index_order_bac, index_order_cab, {{i, j, k}}) ==
              {{k, j, i}});
        CHECK(Alll_Rhs_cab.compute_rhs_tensor_index<3>(
                  index_order_bca, index_order_cab, {{i, j, k}}) ==
              {{j, k, i}});
        CHECK(Alll_Rhs_cab.compute_rhs_tensor_index<3>(
                  index_order_cab, index_order_cab, {{i, j, k}}) ==
              {{i, j, k}});
        CHECK(Alll_Rhs_cab.compute_rhs_tensor_index<3>(
                  index_order_cba, index_order_cab, {{i, j, k}}) ==
              {{i, k, j}});

        // for RHS = {c, b, a}
        CHECK(Alll_Rhs_cba.compute_rhs_tensor_index<3>(
                  index_order_abc, index_order_cba, {{i, j, k}}) ==
              {{k, j, i}});
        CHECK(Alll_Rhs_cba.compute_rhs_tensor_index<3>(
                  index_order_acb, index_order_cba, {{i, j, k}}) ==
              {{j, k, i}});
        CHECK(Alll_Rhs_cba.compute_rhs_tensor_index<3>(
                  index_order_bac, index_order_cba, {{i, j, k}}) ==
              {{k, i, j}});
        CHECK(Alll_Rhs_cba.compute_rhs_tensor_index<3>(
                  index_order_bca, index_order_cba, {{i, j, k}}) ==
              {{j, i, k}});
        CHECK(Alll_Rhs_cba.compute_rhs_tensor_index<3>(
                  index_order_cab, index_order_cba, {{i, j, k}}) ==
              {{i, k, j}});
        CHECK(Alll_Rhs_cba.compute_rhs_tensor_index<3>(
                  index_order_cba, index_order_cba, {{i, j, k}}) ==
              {{i, j, k}});
      }
    }
  }

  /*using Symmetry = Symmetry<1, 1>;
  using Rhs_Index_List = index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
  using Rhs_Tensor = Tensor<double, Symmetry, Rhs_Index_List>;

  using Lhs_Structure = */
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
