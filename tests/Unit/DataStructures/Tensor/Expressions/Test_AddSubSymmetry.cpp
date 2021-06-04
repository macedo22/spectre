// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/Test_AddSubSymmetryImpl.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <auto&... TensorIndices>
using make_tensorindex_list =
    tmpl::list<std::decay_t<decltype(TensorIndices)>...>;

void test_rank0() {
  using symm = Symmetry<>;
  using tensorindex_list = make_tensorindex_list<>;
  using result_symmetry =
      typename detail::AddSubtractSymmetry<symm, symm, tensorindex_list,
                                           tensorindex_list>::type;
  static_assert(
      std::is_same_v<result_symmetry, tmpl::integral_list<std::int32_t>>,
      "Failed AddSubSymmetry rank 0 test case.");
}

void test_rank1() {
  using symm = Symmetry<1>;
  using tensorindex_list = make_tensorindex_list<ti_a>;
  using result_symmetry =
      typename detail::AddSubtractSymmetry<symm, symm, tensorindex_list,
                                           tensorindex_list>::type;
  static_assert(
      std::is_same_v<result_symmetry, tmpl::integral_list<std::int32_t, 1>>,
      "Failed AddSubSymmetry rank 1 test case.");
}

void test_rank2() {
  using symmetric_symm = Symmetry<1, 1>;
  using asymmetric_symm = Symmetry<2, 1>;
  using tensorindex_list_ij = make_tensorindex_list<ti_i, ti_j>;
  using tensorindex_list_ji = make_tensorindex_list<ti_j, ti_i>;

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         symmetric_symm, symmetric_symm, tensorindex_list_ij,
                         tensorindex_list_ij>::type,
                     tmpl::integral_list<std::int32_t, 1, 1>>,
      "Failed AddSubSymmetry rank 2 test case.");

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         symmetric_symm, symmetric_symm, tensorindex_list_ij,
                         tensorindex_list_ji>::type,
                     tmpl::integral_list<std::int32_t, 1, 1>>,
      "Failed AddSubSymmetry rank 2 test case.");

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         asymmetric_symm, symmetric_symm, tensorindex_list_ij,
                         tensorindex_list_ij>::type,
                     tmpl::integral_list<std::int32_t, 2, 1>>,
      "Failed AddSubSymmetry rank 2 test case.");

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         asymmetric_symm, symmetric_symm, tensorindex_list_ij,
                         tensorindex_list_ji>::type,
                     tmpl::integral_list<std::int32_t, 2, 1>>,
      "Failed AddSubSymmetry rank 2 test case.");

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         symmetric_symm, asymmetric_symm, tensorindex_list_ij,
                         tensorindex_list_ij>::type,
                     tmpl::integral_list<std::int32_t, 2, 1>>,
      "Failed AddSubSymmetry rank 2 test case.");

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         symmetric_symm, asymmetric_symm, tensorindex_list_ij,
                         tensorindex_list_ji>::type,
                     tmpl::integral_list<std::int32_t, 2, 1>>,
      "Failed AddSubSymmetry rank 2 test case.");
}

void test_rank3() {
  using symm_111 = Symmetry<1, 1, 1>;
  using symm_121 = Symmetry<1, 2, 1>;
  using symm_211 = Symmetry<2, 1, 1>;
  using symm_221 = Symmetry<2, 2, 1>;
  using symm_321 = Symmetry<3, 2, 1>;

  using tensorindex_list_abc = make_tensorindex_list<ti_a, ti_b, ti_c>;
  using tensorindex_list_acb = make_tensorindex_list<ti_a, ti_c, ti_b>;
  using tensorindex_list_bac = make_tensorindex_list<ti_b, ti_a, ti_c>;
  using tensorindex_list_bca = make_tensorindex_list<ti_b, ti_c, ti_a>;
  using tensorindex_list_cab = make_tensorindex_list<ti_c, ti_a, ti_b>;
  using tensorindex_list_cba = make_tensorindex_list<ti_c, ti_b, ti_a>;

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_111, symm_121, tensorindex_list_abc,
                                   tensorindex_list_bca>::type,
                               tmpl::integral_list<std::int32_t, 2, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_121, symm_111, tensorindex_list_abc,
                                   tensorindex_list_bca>::type,
                               tmpl::integral_list<std::int32_t, 1, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_111, symm_221, tensorindex_list_abc,
                                   tensorindex_list_acb>::type,
                               tmpl::integral_list<std::int32_t, 1, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_221, symm_111, tensorindex_list_abc,
                                   tensorindex_list_acb>::type,
                               tmpl::integral_list<std::int32_t, 2, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_121, symm_221, tensorindex_list_abc,
                                   tensorindex_list_cab>::type,
                               tmpl::integral_list<std::int32_t, 1, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_221, symm_121, tensorindex_list_abc,
                                   tensorindex_list_cab>::type,
                               tmpl::integral_list<std::int32_t, 3, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_221, symm_121, tensorindex_list_cab,
                                   tensorindex_list_abc>::type,
                               tmpl::integral_list<std::int32_t, 2, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_121, symm_221, tensorindex_list_cab,
                                   tensorindex_list_abc>::type,
                               tmpl::integral_list<std::int32_t, 3, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_121, symm_221, tensorindex_list_abc,
                                   tensorindex_list_acb>::type,
                               tmpl::integral_list<std::int32_t, 1, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_221, symm_121, tensorindex_list_abc,
                                   tensorindex_list_acb>::type,
                               tmpl::integral_list<std::int32_t, 2, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_111, symm_321, tensorindex_list_abc,
                                   tensorindex_list_bac>::type,
                               tmpl::integral_list<std::int32_t, 3, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_321, symm_111, tensorindex_list_abc,
                                   tensorindex_list_bac>::type,
                               tmpl::integral_list<std::int32_t, 3, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_211, symm_321, tensorindex_list_abc,
                                   tensorindex_list_cba>::type,
                               tmpl::integral_list<std::int32_t, 3, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");

  static_assert(std::is_same_v<typename detail::AddSubtractSymmetry<
                                   symm_321, symm_211, tensorindex_list_abc,
                                   tensorindex_list_cba>::type,
                               tmpl::integral_list<std::int32_t, 3, 2, 1>>,
                "Failed AddSubSymmetry rank 3 test case.");
}

void test_high_rank() {
  using tensorindex_list = make_tensorindex_list<ti_a, ti_b, ti_c, ti_d, ti_e>;

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         Symmetry<2, 1, 1, 1, 1>, Symmetry<3, 2, 2, 1, 1>,
                         tensorindex_list, tensorindex_list>::type,
                     tmpl::integral_list<std::int32_t, 3, 2, 2, 1, 1>>,
      "Failed AddSubSymmetry high rank test case.");

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         Symmetry<3, 2, 2, 1, 1>, Symmetry<2, 1, 1, 1, 1>,
                         tensorindex_list, tensorindex_list>::type,
                     tmpl::integral_list<std::int32_t, 3, 2, 2, 1, 1>>,
      "Failed AddSubSymmetry high rank test case.");

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         Symmetry<1, 1, 2, 1, 1>, Symmetry<4, 3, 1, 2, 1>,
                         tensorindex_list, tensorindex_list>::type,
                     tmpl::integral_list<std::int32_t, 5, 4, 3, 2, 1>>,
      "Failed AddSubSymmetry high rank test case.");

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         Symmetry<4, 3, 1, 2, 1>, Symmetry<1, 1, 2, 1, 1>,
                         tensorindex_list, tensorindex_list>::type,
                     tmpl::integral_list<std::int32_t, 5, 4, 3, 2, 1>>,
      "Failed AddSubSymmetry high rank test case.");

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         Symmetry<1, 2, 2, 2, 1>, Symmetry<1, 2, 1, 1, 1>,
                         tensorindex_list, tensorindex_list>::type,
                     tmpl::integral_list<std::int32_t, 1, 3, 2, 2, 1>>,
      "Failed AddSubSymmetry high rank test case.");

  static_assert(
      std::is_same_v<typename detail::AddSubtractSymmetry<
                         Symmetry<1, 2, 1, 1, 1>, Symmetry<1, 2, 2, 2, 1>,
                         tensorindex_list, tensorindex_list>::type,
                     tmpl::integral_list<std::int32_t, 1, 3, 2, 2, 1>>,
      "Failed AddSubSymmetry high rank test case.");
}
} // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubSymmetry",
                  "[DataStructures][Unit]") {
  test_rank0();
  test_rank1();
  test_rank2();
  test_rank3();
  test_high_rank();
}
