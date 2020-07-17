// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>
#include <iostream> // REMOVE

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.NewGet",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);
  /*Tensor<double, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Hll{};
  std::iota(Hll.begin(), Hll.end(), 0.0);*/

  //auto All_Exp = All(ti_a, ti_b);

  auto result1 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_a, ti_b));

  auto result2 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_b, ti_a));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      CHECK(result1.get(i, j) == All.get(i, j));
      CHECK(result2.get(j, i) == All.get(i, j));
      CHECK(result1.get(j, i) == result2.get(i, j));
    }
  }

  /*using LhsStructure = Tensor_detail::Structure<Symmetry<1, 1>,
                           SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                           SpatialIndex<3, UpLo::Lo, Frame::Grid>>;
  for (int i = 0; i < 4; ++i) {
      std::cout << All_Exp.template get<LhsStructure>(i) << std::endl;
  }*/

  CHECK(2 == 2);

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
