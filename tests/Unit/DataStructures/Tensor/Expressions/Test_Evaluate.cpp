// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"

namespace {
template <auto&... TensorIndices>
void test_contains_indices_to_contract_impl(const bool expected) {
  CHECK(tenex::detail::contains_indices_to_contract<sizeof...(TensorIndices)>(
            {{std::decay_t<decltype(TensorIndices)>::value...}}) == expected);
}

void test_contains_indices_to_contract() {
  test_contains_indices_to_contract_impl<ti::a, ti::b, ti::c>(false);
  test_contains_indices_to_contract_impl<ti::I, ti::j>(false);
  test_contains_indices_to_contract_impl<ti::j>(false);
  test_contains_indices_to_contract_impl(false);

  test_contains_indices_to_contract_impl<ti::d, ti::D>(true);
  test_contains_indices_to_contract_impl<ti::I, ti::i>(true);
  test_contains_indices_to_contract_impl<ti::a, ti::K, ti::B, ti::b>(true);
  test_contains_indices_to_contract_impl<ti::j, ti::c, ti::J, ti::A, ti::a>(
      true);
}

void test_lhs_tensorindex_reorder_symm_consistency() {
  // TODO : update this message
  const std::string error_msg =
      "tenex::detail::get_reordered_tensorindex_values() assumes a canonical "
      "form for Symmetry that is no longer the actual canonical form of "
      "Symmetry. To make tenex::detail::get_reordered_tensorindex_values() "
      "agree with the current canonical form for Symmetry, the logic of "
      "tenex::detail::get_reordered_tensorindex_values() must be updated";

  if (not std::is_same_v<Symmetry<>, tmpl::integral_list<std::int32_t>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<4>, tmpl::integral_list<std::int32_t, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<1, 2>,
                         tmpl::integral_list<std::int32_t, 2, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<3, 5>,
                         tmpl::integral_list<std::int32_t, 2, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<2, 2, 2>,
                         tmpl::integral_list<std::int32_t, 1, 1, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<8, 4, 5, 5, 8>,
                         tmpl::integral_list<std::int32_t, 1, 3, 2, 2, 1>>) {
    ERROR(error_msg);
  }
}

template <typename LhsTensorIndices, typename ExpectedReorderedTensorIndices>
struct test_impl;

template <typename... LhsTensorIndices,
          typename... ExpectedReorderedTensorIndices>
struct test_impl<tmpl::list<LhsTensorIndices...>,
                 tmpl::list<ExpectedReorderedTensorIndices...>> {
  static constexpr size_t num_indices = sizeof...(LhsTensorIndices);
  static constexpr std::array<size_t, num_indices>
      expected_reordered_tensorindex_values = {
          {ExpectedReorderedTensorIndices::value...}};
  static void apply(const std::array<std::int32_t, num_indices>& symmetry) {
    CHECK(tenex::detail::get_reordered_tensorindex_values<LhsTensorIndices...>(
              symmetry) == expected_reordered_tensorindex_values);
  }
};

void test_lhs_tensorindex_reorder_rank0() {
  std::array<std::int32_t, 0> symmetry{{}};

  using empty_list = make_tensorindex_list<>;

  test_impl<empty_list, empty_list>::apply(symmetry);
}

void test_lhs_tensorindex_reorder_rank1() {
  std::array<std::int32_t, 1> symmetry{{1}};

  using i_list = make_tensorindex_list<ti::i>;
  using a_list = make_tensorindex_list<ti::a>;
  using t_list = make_tensorindex_list<ti::t>;

  using I_list = make_tensorindex_list<ti::I>;
  using A_list = make_tensorindex_list<ti::A>;
  using T_list = make_tensorindex_list<ti::T>;

  // lower
  test_impl<i_list, i_list>::apply(symmetry);
  test_impl<a_list, a_list>::apply(symmetry);
  test_impl<t_list, t_list>::apply(symmetry);

  // upper
  test_impl<I_list, I_list>::apply(symmetry);
  test_impl<A_list, A_list>::apply(symmetry);
  test_impl<T_list, T_list>::apply(symmetry);
}

void test_lhs_tensorindex_reorder_rank2() {
  constexpr size_t num_indices = 2;
  std::array<std::int32_t, num_indices> asymmetric_symm{{2, 1}};
  std::array<std::int32_t, num_indices> symmetric_symm{{1, 1}};

  using ij_list = make_tensorindex_list<ti::i, ti::j>;
  using ji_list = make_tensorindex_list<ti::j, ti::i>;
  using ab_list = make_tensorindex_list<ti::a, ti::b>;
  using ba_list = make_tensorindex_list<ti::b, ti::a>;
  using ia_list = make_tensorindex_list<ti::i, ti::a>;
  using ai_list = make_tensorindex_list<ti::a, ti::i>;
  using it_list = make_tensorindex_list<ti::i, ti::t>;
  using ti_list = make_tensorindex_list<ti::t, ti::i>;
  using at_list = make_tensorindex_list<ti::a, ti::t>;
  using ta_list = make_tensorindex_list<ti::t, ti::a>;
  using tt_list = make_tensorindex_list<ti::t, ti::t>;

  using IJ_list = make_tensorindex_list<ti::i, ti::j>;
  using JI_list = make_tensorindex_list<ti::j, ti::i>;
  using AB_list = make_tensorindex_list<ti::a, ti::b>;
  using BA_list = make_tensorindex_list<ti::b, ti::a>;
  using IA_list = make_tensorindex_list<ti::i, ti::a>;
  using AI_list = make_tensorindex_list<ti::a, ti::i>;
  using IT_list = make_tensorindex_list<ti::i, ti::t>;
  using TI_list = make_tensorindex_list<ti::t, ti::i>;
  using AT_list = make_tensorindex_list<ti::a, ti::t>;
  using TA_list = make_tensorindex_list<ti::t, ti::a>;
  using TT_list = make_tensorindex_list<ti::t, ti::t>;

  using iJ_list = make_tensorindex_list<ti::i, ti::j>;
  using jI_list = make_tensorindex_list<ti::j, ti::i>;
  using aB_list = make_tensorindex_list<ti::a, ti::b>;
  using bA_list = make_tensorindex_list<ti::b, ti::a>;
  using iA_list = make_tensorindex_list<ti::i, ti::a>;
  using aI_list = make_tensorindex_list<ti::a, ti::i>;
  using iT_list = make_tensorindex_list<ti::i, ti::t>;
  using tI_list = make_tensorindex_list<ti::t, ti::i>;
  using aT_list = make_tensorindex_list<ti::a, ti::t>;
  using tA_list = make_tensorindex_list<ti::t, ti::a>;
  using tT_list = make_tensorindex_list<ti::t, ti::t>;

  using Ij_list = make_tensorindex_list<ti::i, ti::j>;
  using Ji_list = make_tensorindex_list<ti::j, ti::i>;
  using Ab_list = make_tensorindex_list<ti::a, ti::b>;
  using Ba_list = make_tensorindex_list<ti::b, ti::a>;
  using Ia_list = make_tensorindex_list<ti::i, ti::a>;
  using Ai_list = make_tensorindex_list<ti::a, ti::i>;
  using It_list = make_tensorindex_list<ti::i, ti::t>;
  using Ti_list = make_tensorindex_list<ti::t, ti::i>;
  using At_list = make_tensorindex_list<ti::a, ti::t>;
  using Ta_list = make_tensorindex_list<ti::t, ti::a>;
  using Tt_list = make_tensorindex_list<ti::t, ti::t>;

  // tnsr::aa<double, 3> R{0.0};
  // tnsr::aa<double, 3> L{};

  // tenex::evaluate<ti::i, ti::t>(make_not_null(&L), R(ti::i, ti::t));

  // lower
  test_impl<ij_list, ij_list>::apply(asymmetric_symm);
  test_impl<ij_list, ij_list>::apply(symmetric_symm);

  test_impl<ji_list, ji_list>::apply(asymmetric_symm);
  test_impl<ji_list, ij_list>::apply(symmetric_symm);

  test_impl<ab_list, ab_list>::apply(asymmetric_symm);
  test_impl<ab_list, ab_list>::apply(symmetric_symm);

  test_impl<ba_list, ba_list>::apply(asymmetric_symm);
  test_impl<ba_list, ab_list>::apply(symmetric_symm);

  test_impl<ai_list, ai_list>::apply(asymmetric_symm);
  test_impl<ai_list, ia_list>::apply(symmetric_symm);

  test_impl<ia_list, ia_list>::apply(asymmetric_symm);
  test_impl<ia_list, ia_list>::apply(symmetric_symm);

  test_impl<it_list, it_list>::apply(asymmetric_symm);
  test_impl<it_list, it_list>::apply(symmetric_symm);

  test_impl<ti_list, ti_list>::apply(asymmetric_symm);
  test_impl<ti_list, it_list>::apply(symmetric_symm);

  test_impl<at_list, at_list>::apply(asymmetric_symm);
  test_impl<at_list, at_list>::apply(symmetric_symm);

  test_impl<ta_list, ta_list>::apply(asymmetric_symm);
  test_impl<ta_list, at_list>::apply(symmetric_symm);

  test_impl<tt_list, tt_list>::apply(asymmetric_symm);
  test_impl<tt_list, tt_list>::apply(symmetric_symm);

  // upper
  test_impl<IJ_list, IJ_list>::apply(asymmetric_symm);
  test_impl<IJ_list, IJ_list>::apply(symmetric_symm);

  test_impl<JI_list, JI_list>::apply(asymmetric_symm);
  test_impl<JI_list, IJ_list>::apply(symmetric_symm);

  test_impl<AB_list, AB_list>::apply(asymmetric_symm);
  test_impl<AB_list, AB_list>::apply(symmetric_symm);

  test_impl<BA_list, BA_list>::apply(asymmetric_symm);
  test_impl<BA_list, AB_list>::apply(symmetric_symm);

  test_impl<AI_list, AI_list>::apply(asymmetric_symm);
  test_impl<AI_list, IA_list>::apply(symmetric_symm);

  test_impl<IA_list, IA_list>::apply(asymmetric_symm);
  test_impl<IA_list, IA_list>::apply(symmetric_symm);

  test_impl<IT_list, IT_list>::apply(asymmetric_symm);
  test_impl<IT_list, IT_list>::apply(symmetric_symm);

  test_impl<TI_list, TI_list>::apply(asymmetric_symm);
  test_impl<TI_list, IT_list>::apply(symmetric_symm);

  test_impl<AT_list, AT_list>::apply(asymmetric_symm);
  test_impl<AT_list, AT_list>::apply(symmetric_symm);

  test_impl<TA_list, TA_list>::apply(asymmetric_symm);
  test_impl<TA_list, AT_list>::apply(symmetric_symm);

  test_impl<TT_list, TT_list>::apply(asymmetric_symm);
  test_impl<TT_list, TT_list>::apply(symmetric_symm);

  // lower upper
  test_impl<iJ_list, iJ_list>::apply(asymmetric_symm);

  test_impl<jI_list, jI_list>::apply(asymmetric_symm);

  test_impl<aB_list, aB_list>::apply(asymmetric_symm);

  test_impl<bA_list, bA_list>::apply(asymmetric_symm);

  test_impl<aI_list, aI_list>::apply(asymmetric_symm);

  test_impl<iA_list, iA_list>::apply(asymmetric_symm);

  test_impl<iT_list, iT_list>::apply(asymmetric_symm);

  test_impl<tI_list, tI_list>::apply(asymmetric_symm);

  test_impl<aT_list, aT_list>::apply(asymmetric_symm);

  test_impl<tA_list, tA_list>::apply(asymmetric_symm);

  test_impl<tT_list, tT_list>::apply(asymmetric_symm);

  // upper lower
  test_impl<Ij_list, Ij_list>::apply(asymmetric_symm);

  test_impl<Ji_list, Ji_list>::apply(asymmetric_symm);

  test_impl<Ab_list, Ab_list>::apply(asymmetric_symm);

  test_impl<Ba_list, Ba_list>::apply(asymmetric_symm);

  test_impl<Ai_list, Ai_list>::apply(asymmetric_symm);

  test_impl<Ia_list, Ia_list>::apply(asymmetric_symm);

  test_impl<It_list, It_list>::apply(asymmetric_symm);

  test_impl<Ti_list, Ti_list>::apply(asymmetric_symm);

  test_impl<At_list, At_list>::apply(asymmetric_symm);

  test_impl<Ta_list, Ta_list>::apply(asymmetric_symm);

  test_impl<Tt_list, Tt_list>::apply(asymmetric_symm);
}

// void test_lhs_tensorindex_reorder_rank3() {
//   using symm_111 = Symmetry<1, 1, 1>;
//   using symm_121 = Symmetry<1, 2, 1>;
//   using symm_211 = Symmetry<2, 1, 1>;
//   using symm_221 = Symmetry<2, 2, 1>;
//   using symm_321 = Symmetry<3, 2, 1>;

//   using tensorindex_list_abc = make_tensorindex_list<ti::a, ti::b, ti::c>;
//   using tensorindex_list_acb = make_tensorindex_list<ti::a, ti::c, ti::b>;
//   using tensorindex_list_bac = make_tensorindex_list<ti::b, ti::a, ti::c>;
//   using tensorindex_list_bca = make_tensorindex_list<ti::b, ti::c, ti::a>;
//   using tensorindex_list_cab = make_tensorindex_list<ti::c, ti::a, ti::b>;
//   using tensorindex_list_cba = make_tensorindex_list<ti::c, ti::b, ti::a>;

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_111, symm_121, tensorindex_list_abc,
//                            tensorindex_list_bca>::type,
//                        tmpl::integral_list<std::int32_t, 2, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_121, symm_111, tensorindex_list_abc,
//                            tensorindex_list_bca>::type,
//                        tmpl::integral_list<std::int32_t, 1, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_111, symm_221, tensorindex_list_abc,
//                            tensorindex_list_acb>::type,
//                        tmpl::integral_list<std::int32_t, 1, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_221, symm_111, tensorindex_list_abc,
//                            tensorindex_list_acb>::type,
//                        tmpl::integral_list<std::int32_t, 2, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_121, symm_221, tensorindex_list_abc,
//                            tensorindex_list_cab>::type,
//                        tmpl::integral_list<std::int32_t, 1, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_221, symm_121, tensorindex_list_abc,
//                            tensorindex_list_cab>::type,
//                        tmpl::integral_list<std::int32_t, 3, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_221, symm_121, tensorindex_list_cab,
//                            tensorindex_list_abc>::type,
//                        tmpl::integral_list<std::int32_t, 2, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_121, symm_221, tensorindex_list_cab,
//                            tensorindex_list_abc>::type,
//                        tmpl::integral_list<std::int32_t, 3, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_121, symm_221, tensorindex_list_abc,
//                            tensorindex_list_acb>::type,
//                        tmpl::integral_list<std::int32_t, 1, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_221, symm_121, tensorindex_list_abc,
//                            tensorindex_list_acb>::type,
//                        tmpl::integral_list<std::int32_t, 2, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_111, symm_321, tensorindex_list_abc,
//                            tensorindex_list_bac>::type,
//                        tmpl::integral_list<std::int32_t, 3, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_321, symm_111, tensorindex_list_abc,
//                            tensorindex_list_bac>::type,
//                        tmpl::integral_list<std::int32_t, 3, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_211, symm_321, tensorindex_list_abc,
//                            tensorindex_list_cba>::type,
//                        tmpl::integral_list<std::int32_t, 3, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            symm_321, symm_211, tensorindex_list_abc,
//                            tensorindex_list_cba>::type,
//                        tmpl::integral_list<std::int32_t, 3, 2, 1>>);
// }

// void test_lhs_tensorindex_reorder_high_rank() {
//   using tensorindex_list =
//       make_tensorindex_list<ti::a, ti::b, ti::c, ti::d, ti::e>;

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            Symmetry<2, 1, 1, 1, 1>, Symmetry<3, 2, 2, 1, 1>,
//                            tensorindex_list, tensorindex_list>::type,
//                        tmpl::integral_list<std::int32_t, 3, 2, 2, 1, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            Symmetry<3, 2, 2, 1, 1>, Symmetry<2, 1, 1, 1, 1>,
//                            tensorindex_list, tensorindex_list>::type,
//                        tmpl::integral_list<std::int32_t, 3, 2, 2, 1, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            Symmetry<1, 1, 2, 1, 1>, Symmetry<4, 3, 1, 2, 1>,
//                            tensorindex_list, tensorindex_list>::type,
//                        tmpl::integral_list<std::int32_t, 5, 4, 3, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            Symmetry<4, 3, 1, 2, 1>, Symmetry<1, 1, 2, 1, 1>,
//                            tensorindex_list, tensorindex_list>::type,
//                        tmpl::integral_list<std::int32_t, 5, 4, 3, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            Symmetry<1, 2, 2, 2, 1>, Symmetry<1, 2, 1, 1, 1>,
//                            tensorindex_list, tensorindex_list>::type,
//                        tmpl::integral_list<std::int32_t, 1, 3, 2, 2, 1>>);

//   CHECK(std::is_same_v<typename tenex::detail::AddSubSymmetry<
//                            Symmetry<1, 2, 1, 1, 1>, Symmetry<1, 2, 2, 2, 1>,
//                            tensorindex_list, tensorindex_list>::type,
//                        tmpl::integral_list<std::int32_t, 1, 3, 2, 2, 1>>);
// }

void test_lhs_tensorindex_reorder() {
  //   test_lhs_tensorindex_reorder_symm_consistency();
  test_lhs_tensorindex_reorder_rank0();
  test_lhs_tensorindex_reorder_rank1();
  test_lhs_tensorindex_reorder_rank2();
  //   test_lhs_tensorindex_reorder_rank3();
  //   test_lhs_tensorindex_reorder_high_rank();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Evaluate",
                  "[DataStructures][Unit]") {
  //   test_contains_indices_to_contract();
  test_lhs_tensorindex_reorder();
}
