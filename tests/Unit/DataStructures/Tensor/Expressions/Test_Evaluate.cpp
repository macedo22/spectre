// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using symm_datatype = typename Tensor_detail::symmetry_datatype;

template <symm_datatype... Is>
using expected_symm = tmpl::integral_list<symm_datatype, Is...>;

template <auto&... TensorIndices>
void test_contains_indices_to_contract_impl(const bool expected) {
  CHECK(tenex::detail::contains_indices_to_contract<sizeof...(TensorIndices)>(
            {{std::decay_t<decltype(TensorIndices)>::value...}}) == expected);
}

// Tests the helper function `tenex::detail::contains_indices_to_contract`
// correctly determines whether or not a list of tensor indices contains at
// least one index pair to contract
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

// Tests that the canonical ordering of symmetry values by `Symmetry` is
// consistent with what `tenex::detail::get_reordered_tensorindex_values`
// expects, which is that the symmetry values assigned to indepdenent indices
// is ascending from the rightmost position moving leftward with the rightmost
// symmetry value starting at 1
void test_lhs_tensorindex_reorder_symm_consistency() {
  const std::string error_msg =
      "tenex::detail::get_reordered_tensorindex_values() assumes a canonical "
      "form for Symmetry that is no longer the actual canonical form of "
      "Symmetry. The logic of this unit test and "
      "tenex::detail::get_reordered_tensorindex_values() must be updated to "
      "agree with the current canonical form for Symmetry";

  if (not std::is_same_v<Symmetry<>, expected_symm<>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<4>, expected_symm<1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<1, 2>, expected_symm<2, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<3, 5>, expected_symm<2, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<2, 2, 2>, expected_symm<1, 1, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<8, 4, 5, 5, 8>,
                         expected_symm<1, 3, 2, 2, 1>>) {
    ERROR(error_msg);
  }
}

// Tests that the canonical ordering of multi-indices by
// `Tensor_detail::Structure` is consistent with what
// `tenex::detail::evaluate_impl` expects. Specifically, it checks that the
// canonical multi-indices of independent tensor components are ordered such
// that within each subset of symmetric indices, the index values are
// ascending from the rightmost index to the left, which is what `evaluate_impl`
// assumes.
void test_evaluate_and_canon_multi_index_consistency() {
  const std::string error_msg =
      "tenex::evaluate() assumes a canonical form for multi-indices that is no "
      "longer consistent with the canonical form defined by "
      "Tensor_detail::Structure. The logic of this unit test and "
      "tenex::detail::evaluate_impl must be updated to agree with the current "
      "canonical form for multi-indices.";

  using datatype = double;
  using frame = Frame::Inertial;

  using iii = tnsr::iii<datatype, 3>::structure;
  using aaa = Tensor<datatype, Symmetry<1, 1, 1>,
                     index_list<SpacetimeIndex<3, UpLo::Lo, frame>,
                                SpacetimeIndex<3, UpLo::Lo, frame>,
                                SpacetimeIndex<3, UpLo::Lo, frame>>>::structure;
  using iaai = Tensor<datatype, Symmetry<1, 2, 2, 1>,
                      index_list<SpatialIndex<3, UpLo::Lo, frame>,
                                 SpacetimeIndex<3, UpLo::Lo, frame>,
                                 SpacetimeIndex<3, UpLo::Lo, frame>,
                                 SpatialIndex<3, UpLo::Lo, frame>>>::structure;
  using iiaa =
      Tensor<datatype, Symmetry<2, 2, 1, 1>,
             index_list<SpatialIndex<2, UpLo::Lo, frame>,
                        SpatialIndex<2, UpLo::Lo, frame>,
                        SpacetimeIndex<2, UpLo::Lo, frame>,
                        SpacetimeIndex<2, UpLo::Lo, frame>>>::structure;
  using aiai =
      Tensor<datatype, Symmetry<2, 1, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, frame>,
                        SpatialIndex<3, UpLo::Lo, frame>,
                        SpacetimeIndex<3, UpLo::Lo, frame>,
                        SpatialIndex<3, UpLo::Lo, frame>>>::structure;
  using iiii =
      Tensor<datatype, Symmetry<1, 1, 1, 1>,
             index_list<SpatialIndex<3, UpLo::Lo, frame>,
                        SpatialIndex<3, UpLo::Lo, frame>,
                        SpatialIndex<3, UpLo::Lo, frame>,
                        SpatialIndex<3, UpLo::Lo, frame>>>::structure;

  for (size_t i = 0; i < iii::size(); i++) {
    const auto canon_multi_index = iii::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[1]);
    CHECK(canon_multi_index[1] >= canon_multi_index[2]);
  }

  for (size_t i = 0; i < aaa::size(); i++) {
    const auto canon_multi_index = aaa::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[1]);
    CHECK(canon_multi_index[1] >= canon_multi_index[2]);
  }

  for (size_t i = 0; i < iaai::size(); i++) {
    const auto canon_multi_index = iaai::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[3]);
    CHECK(canon_multi_index[1] >= canon_multi_index[2]);
  }

  for (size_t i = 0; i < iiaa::size(); i++) {
    const auto canon_multi_index = iiaa::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[1]);
    CHECK(canon_multi_index[2] >= canon_multi_index[3]);
  }

  for (size_t i = 0; i < aiai::size(); i++) {
    const auto canon_multi_index = aiai::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[2]);
    CHECK(canon_multi_index[1] >= canon_multi_index[3]);
  }

  for (size_t i = 0; i < iiii::size(); i++) {
    const auto canon_multi_index = iiii::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[1]);
    CHECK(canon_multi_index[1] >= canon_multi_index[2]);
    CHECK(canon_multi_index[2] >= canon_multi_index[3]);
  }
}

// Tests that the canonical ordering of a list of `TensorIndex`s done by
// `tenex::detail::get_reordered_tensorindex_values`
template <typename LhsTensorIndices, typename ExpectedReorderedTensorIndices>
struct test_lhs_tensorindex_reorder_impl;

template <typename... LhsTensorIndices,
          typename... ExpectedReorderedTensorIndices>
struct test_lhs_tensorindex_reorder_impl<
    tmpl::list<LhsTensorIndices...>,
    tmpl::list<ExpectedReorderedTensorIndices...>> {
  static constexpr size_t num_indices = sizeof...(LhsTensorIndices);
  static constexpr std::array<size_t, num_indices>
      expected_reordered_tensorindex_values = {
          {ExpectedReorderedTensorIndices::value...}};
  static void apply(const std::array<symm_datatype, num_indices>& symmetry) {
    CHECK(tenex::detail::get_reordered_tensorindex_values<LhsTensorIndices...>(
              symmetry) == expected_reordered_tensorindex_values);
  }
};

// Tests that the canonical ordering of a list of `TensorIndex`s done by
// `tenex::detail::get_reordered_tensorindex_values` for a rank 0 tensor
void test_lhs_tensorindex_reorder_rank0() {
  const std::array<symm_datatype, 0> symmetry{{}};

  using empty_list = make_tensorindex_list<>;

  test_lhs_tensorindex_reorder_impl<empty_list, empty_list>::apply(symmetry);
}

// Tests that the canonical ordering of a list of `TensorIndex`s done by
// `tenex::detail::get_reordered_tensorindex_values` for a rank 1 tensor
void test_lhs_tensorindex_reorder_rank1() {
  const std::array<symm_datatype, 1> symmetry{{1}};

  using i_list = make_tensorindex_list<ti::i>;
  using a_list = make_tensorindex_list<ti::a>;
  using t_list = make_tensorindex_list<ti::t>;

  using I_list = make_tensorindex_list<ti::I>;
  using A_list = make_tensorindex_list<ti::A>;
  using T_list = make_tensorindex_list<ti::T>;

  // lower
  test_lhs_tensorindex_reorder_impl<i_list, i_list>::apply(symmetry);
  test_lhs_tensorindex_reorder_impl<a_list, a_list>::apply(symmetry);
  test_lhs_tensorindex_reorder_impl<t_list, t_list>::apply(symmetry);

  // upper
  test_lhs_tensorindex_reorder_impl<I_list, I_list>::apply(symmetry);
  test_lhs_tensorindex_reorder_impl<A_list, A_list>::apply(symmetry);
  test_lhs_tensorindex_reorder_impl<T_list, T_list>::apply(symmetry);
}

// Tests that the canonical ordering of a list of `TensorIndex`s done by
// `tenex::detail::get_reordered_tensorindex_values` for a rank 2 tensor
void test_lhs_tensorindex_reorder_rank2() {
  constexpr size_t num_indices = 2;
  const std::array<symm_datatype, num_indices> asymmetric_symm{{2, 1}};
  const std::array<symm_datatype, num_indices> symmetric_symm{{1, 1}};

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

  // lower
  test_lhs_tensorindex_reorder_impl<ij_list, ij_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<ij_list, ij_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<ji_list, ji_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<ji_list, ij_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<ab_list, ab_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<ab_list, ab_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<ba_list, ba_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<ba_list, ab_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<ai_list, ai_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<ai_list, ia_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<ia_list, ia_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<ia_list, ia_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<it_list, it_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<it_list, it_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<ti_list, ti_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<ti_list, it_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<at_list, at_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<at_list, at_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<ta_list, ta_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<ta_list, at_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<tt_list, tt_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<tt_list, tt_list>::apply(symmetric_symm);

  // upper
  test_lhs_tensorindex_reorder_impl<IJ_list, IJ_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<IJ_list, IJ_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<JI_list, JI_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<JI_list, IJ_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<AB_list, AB_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<AB_list, AB_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<BA_list, BA_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<BA_list, AB_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<AI_list, AI_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<AI_list, IA_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<IA_list, IA_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<IA_list, IA_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<IT_list, IT_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<IT_list, IT_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<TI_list, TI_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<TI_list, IT_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<AT_list, AT_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<AT_list, AT_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<TA_list, TA_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<TA_list, AT_list>::apply(symmetric_symm);

  test_lhs_tensorindex_reorder_impl<TT_list, TT_list>::apply(asymmetric_symm);
  test_lhs_tensorindex_reorder_impl<TT_list, TT_list>::apply(symmetric_symm);

  // lower upper
  test_lhs_tensorindex_reorder_impl<iJ_list, iJ_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<jI_list, jI_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<aB_list, aB_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<bA_list, bA_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<aI_list, aI_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<iA_list, iA_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<iT_list, iT_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<tI_list, tI_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<aT_list, aT_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<tA_list, tA_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<tT_list, tT_list>::apply(asymmetric_symm);

  // upper lower
  test_lhs_tensorindex_reorder_impl<Ij_list, Ij_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<Ji_list, Ji_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<Ab_list, Ab_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<Ba_list, Ba_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<Ai_list, Ai_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<Ia_list, Ia_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<It_list, It_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<Ti_list, Ti_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<At_list, At_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<Ta_list, Ta_list>::apply(asymmetric_symm);

  test_lhs_tensorindex_reorder_impl<Tt_list, Tt_list>::apply(asymmetric_symm);
}

// Tests that the canonical ordering of a list of `TensorIndex`s done by
// `tenex::detail::get_reordered_tensorindex_values` for a rank 3 tensor
void test_lhs_tensorindex_reorder_rank3() {
  constexpr size_t num_indices = 3;
  const std::array<symm_datatype, num_indices> symm_111{{1, 1, 1}};
  const std::array<symm_datatype, num_indices> symm_121{{1, 2, 1}};
  const std::array<symm_datatype, num_indices> symm_211{{2, 1, 1}};
  const std::array<symm_datatype, num_indices> symm_221{{2, 2, 1}};
  const std::array<symm_datatype, num_indices> symm_321{{3, 2, 1}};

  using ijk_list = make_tensorindex_list<ti::i, ti::j, ti::k>;
  using ikj_list = make_tensorindex_list<ti::i, ti::k, ti::j>;
  using jik_list = make_tensorindex_list<ti::j, ti::i, ti::k>;
  using jki_list = make_tensorindex_list<ti::j, ti::k, ti::i>;
  using kij_list = make_tensorindex_list<ti::k, ti::i, ti::j>;
  using kji_list = make_tensorindex_list<ti::k, ti::j, ti::i>;

  // ijk
  test_lhs_tensorindex_reorder_impl<ijk_list, ijk_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<ijk_list, ijk_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<ijk_list, ijk_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<ijk_list, ijk_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<ijk_list, ijk_list>::apply(symm_321);

  // ikj
  test_lhs_tensorindex_reorder_impl<ikj_list, ijk_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<ikj_list, ikj_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<ikj_list, ijk_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<ikj_list, ikj_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<ikj_list, ikj_list>::apply(symm_321);

  // jik
  test_lhs_tensorindex_reorder_impl<jik_list, ijk_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<jik_list, jik_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<jik_list, jik_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<jik_list, ijk_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<jik_list, jik_list>::apply(symm_321);

  // jki
  test_lhs_tensorindex_reorder_impl<jki_list, ijk_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<jki_list, ikj_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<jki_list, jik_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<jki_list, jki_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<jki_list, jki_list>::apply(symm_321);

  // kij
  test_lhs_tensorindex_reorder_impl<kij_list, ijk_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<kij_list, jik_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<kij_list, kij_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<kij_list, ikj_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<kij_list, kij_list>::apply(symm_321);

  // kji
  test_lhs_tensorindex_reorder_impl<kji_list, ijk_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<kji_list, ijk_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<kji_list, kij_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<kji_list, jki_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<kji_list, kji_list>::apply(symm_321);

  using abc_list = make_tensorindex_list<ti::i, ti::j, ti::k>;
  using acb_list = make_tensorindex_list<ti::i, ti::k, ti::j>;
  using bac_list = make_tensorindex_list<ti::j, ti::i, ti::k>;
  using bca_list = make_tensorindex_list<ti::j, ti::k, ti::i>;
  using cab_list = make_tensorindex_list<ti::k, ti::i, ti::j>;
  using cba_list = make_tensorindex_list<ti::k, ti::j, ti::i>;

  // abc
  test_lhs_tensorindex_reorder_impl<abc_list, abc_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<abc_list, abc_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<abc_list, abc_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<abc_list, abc_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<abc_list, abc_list>::apply(symm_321);

  // acb
  test_lhs_tensorindex_reorder_impl<acb_list, abc_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<acb_list, acb_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<acb_list, abc_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<acb_list, acb_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<acb_list, acb_list>::apply(symm_321);

  // bac
  test_lhs_tensorindex_reorder_impl<bac_list, abc_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<bac_list, bac_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<bac_list, bac_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<bac_list, abc_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<bac_list, bac_list>::apply(symm_321);

  // bca
  test_lhs_tensorindex_reorder_impl<bca_list, abc_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<bca_list, acb_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<bca_list, bac_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<bca_list, bca_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<bca_list, bca_list>::apply(symm_321);

  // cab
  test_lhs_tensorindex_reorder_impl<cab_list, abc_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<cab_list, bac_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<cab_list, cab_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<cab_list, acb_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<cab_list, cab_list>::apply(symm_321);

  // cba
  test_lhs_tensorindex_reorder_impl<cba_list, abc_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<cba_list, abc_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<cba_list, cab_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<cba_list, bca_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<cba_list, cba_list>::apply(symm_321);

  using ija_list = make_tensorindex_list<ti::i, ti::j, ti::a>;
  using iaj_list = make_tensorindex_list<ti::i, ti::a, ti::j>;
  using jia_list = make_tensorindex_list<ti::j, ti::i, ti::a>;
  using jai_list = make_tensorindex_list<ti::j, ti::a, ti::i>;
  using aij_list = make_tensorindex_list<ti::a, ti::i, ti::j>;
  using aji_list = make_tensorindex_list<ti::a, ti::j, ti::i>;

  // ija
  test_lhs_tensorindex_reorder_impl<ija_list, ija_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<ija_list, ija_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<ija_list, ija_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<ija_list, ija_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<ija_list, ija_list>::apply(symm_321);

  // iaj
  test_lhs_tensorindex_reorder_impl<iaj_list, ija_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<iaj_list, iaj_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<iaj_list, ija_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<iaj_list, iaj_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<iaj_list, iaj_list>::apply(symm_321);

  // jia
  test_lhs_tensorindex_reorder_impl<jia_list, ija_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<jia_list, jia_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<jia_list, jia_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<jia_list, ija_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<jia_list, jia_list>::apply(symm_321);

  // jai
  test_lhs_tensorindex_reorder_impl<jai_list, ija_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<jai_list, iaj_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<jai_list, jia_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<jai_list, jai_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<jai_list, jai_list>::apply(symm_321);

  // aij
  test_lhs_tensorindex_reorder_impl<aij_list, ija_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<aij_list, jia_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<aij_list, aij_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<aij_list, iaj_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<aij_list, aij_list>::apply(symm_321);

  // aji
  test_lhs_tensorindex_reorder_impl<aji_list, ija_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<aji_list, ija_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<aji_list, aij_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<aji_list, jai_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<aji_list, aji_list>::apply(symm_321);

  using abi_list = make_tensorindex_list<ti::a, ti::b, ti::i>;
  using aib_list = make_tensorindex_list<ti::a, ti::i, ti::b>;
  using bai_list = make_tensorindex_list<ti::b, ti::a, ti::i>;
  using bia_list = make_tensorindex_list<ti::b, ti::i, ti::a>;
  using iab_list = make_tensorindex_list<ti::i, ti::a, ti::b>;
  using iba_list = make_tensorindex_list<ti::i, ti::b, ti::a>;

  // abi
  test_lhs_tensorindex_reorder_impl<abi_list, iab_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<abi_list, iba_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<abi_list, aib_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<abi_list, abi_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<abi_list, abi_list>::apply(symm_321);

  // aib
  test_lhs_tensorindex_reorder_impl<aib_list, iab_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<aib_list, aib_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<aib_list, aib_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<aib_list, iab_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<aib_list, aib_list>::apply(symm_321);

  // bai
  test_lhs_tensorindex_reorder_impl<bai_list, iab_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<bai_list, iab_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<bai_list, bia_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<bai_list, abi_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<bai_list, bai_list>::apply(symm_321);

  // bia
  test_lhs_tensorindex_reorder_impl<bia_list, iab_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<bia_list, aib_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<bia_list, bia_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<bia_list, iba_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<bia_list, bia_list>::apply(symm_321);

  // iab
  test_lhs_tensorindex_reorder_impl<iab_list, iab_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<iab_list, iab_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<iab_list, iab_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<iab_list, iab_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<iab_list, iab_list>::apply(symm_321);

  // iba
  test_lhs_tensorindex_reorder_impl<iba_list, iab_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<iba_list, iba_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<iba_list, iab_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<iba_list, iba_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<iba_list, iba_list>::apply(symm_321);

  using ijt_list = make_tensorindex_list<ti::i, ti::j, ti::t>;
  using itj_list = make_tensorindex_list<ti::i, ti::t, ti::j>;
  using jit_list = make_tensorindex_list<ti::j, ti::i, ti::t>;
  using jti_list = make_tensorindex_list<ti::j, ti::t, ti::i>;
  using tij_list = make_tensorindex_list<ti::t, ti::i, ti::j>;
  using tji_list = make_tensorindex_list<ti::t, ti::j, ti::i>;

  // ijt
  test_lhs_tensorindex_reorder_impl<ijt_list, ijt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<ijt_list, ijt_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<ijt_list, ijt_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<ijt_list, ijt_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<ijt_list, ijt_list>::apply(symm_321);

  // itj
  test_lhs_tensorindex_reorder_impl<itj_list, ijt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<itj_list, itj_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<itj_list, ijt_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<itj_list, itj_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<itj_list, itj_list>::apply(symm_321);

  // jit
  test_lhs_tensorindex_reorder_impl<jit_list, ijt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<jit_list, jit_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<jit_list, jit_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<jit_list, ijt_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<jit_list, jit_list>::apply(symm_321);

  // jti
  test_lhs_tensorindex_reorder_impl<jti_list, ijt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<jti_list, itj_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<jti_list, jit_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<jti_list, jti_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<jti_list, jti_list>::apply(symm_321);

  // tij
  test_lhs_tensorindex_reorder_impl<tij_list, ijt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<tij_list, jit_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<tij_list, tij_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<tij_list, itj_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<tij_list, tij_list>::apply(symm_321);

  // tji
  test_lhs_tensorindex_reorder_impl<tji_list, ijt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<tji_list, ijt_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<tji_list, tij_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<tji_list, jti_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<tji_list, tji_list>::apply(symm_321);

  using abt_list = make_tensorindex_list<ti::i, ti::j, ti::t>;
  using atb_list = make_tensorindex_list<ti::i, ti::t, ti::j>;
  using bat_list = make_tensorindex_list<ti::j, ti::i, ti::t>;
  using bta_list = make_tensorindex_list<ti::j, ti::t, ti::i>;
  using tab_list = make_tensorindex_list<ti::t, ti::i, ti::j>;
  using tba_list = make_tensorindex_list<ti::t, ti::j, ti::i>;

  // abt
  test_lhs_tensorindex_reorder_impl<abt_list, abt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<abt_list, abt_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<abt_list, abt_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<abt_list, abt_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<abt_list, abt_list>::apply(symm_321);

  // atb
  test_lhs_tensorindex_reorder_impl<atb_list, abt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<atb_list, atb_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<atb_list, abt_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<atb_list, atb_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<atb_list, atb_list>::apply(symm_321);

  // bat
  test_lhs_tensorindex_reorder_impl<bat_list, abt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<bat_list, bat_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<bat_list, bat_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<bat_list, abt_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<bat_list, bat_list>::apply(symm_321);

  // bta
  test_lhs_tensorindex_reorder_impl<bta_list, abt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<bta_list, atb_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<bta_list, bat_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<bta_list, bta_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<bta_list, bta_list>::apply(symm_321);

  // tab
  test_lhs_tensorindex_reorder_impl<tab_list, abt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<tab_list, bat_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<tab_list, tab_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<tab_list, atb_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<tab_list, tab_list>::apply(symm_321);

  // tba
  test_lhs_tensorindex_reorder_impl<tba_list, abt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<tba_list, abt_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<tba_list, tab_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<tba_list, bta_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<tba_list, tba_list>::apply(symm_321);

  using itt_list = make_tensorindex_list<ti::a, ti::t, ti::t>;
  using tit_list = make_tensorindex_list<ti::t, ti::a, ti::t>;
  using tti_list = make_tensorindex_list<ti::t, ti::t, ti::a>;

  // itt
  test_lhs_tensorindex_reorder_impl<itt_list, itt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<itt_list, itt_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<itt_list, itt_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<itt_list, itt_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<itt_list, itt_list>::apply(symm_321);

  // tit
  test_lhs_tensorindex_reorder_impl<tit_list, itt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<tit_list, tit_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<tit_list, tit_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<tit_list, itt_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<tit_list, tit_list>::apply(symm_321);

  // tti
  test_lhs_tensorindex_reorder_impl<tti_list, itt_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<tti_list, itt_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<tti_list, tit_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<tti_list, tti_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<tti_list, tti_list>::apply(symm_321);

  using att_list = make_tensorindex_list<ti::a, ti::t, ti::t>;
  using tat_list = make_tensorindex_list<ti::t, ti::a, ti::t>;
  using tta_list = make_tensorindex_list<ti::t, ti::t, ti::a>;

  // att
  test_lhs_tensorindex_reorder_impl<att_list, att_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<att_list, att_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<att_list, att_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<att_list, att_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<att_list, att_list>::apply(symm_321);

  // tat
  test_lhs_tensorindex_reorder_impl<tat_list, att_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<tat_list, tat_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<tat_list, tat_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<tat_list, att_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<tat_list, tat_list>::apply(symm_321);

  // tta
  test_lhs_tensorindex_reorder_impl<tta_list, att_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<tta_list, att_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<tta_list, tat_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<tta_list, tta_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<tta_list, tta_list>::apply(symm_321);

  using iat_list = make_tensorindex_list<ti::i, ti::a, ti::t>;
  using ita_list = make_tensorindex_list<ti::i, ti::t, ti::a>;
  using ait_list = make_tensorindex_list<ti::a, ti::i, ti::t>;
  using ati_list = make_tensorindex_list<ti::a, ti::t, ti::i>;
  using tia_list = make_tensorindex_list<ti::t, ti::i, ti::a>;
  using tai_list = make_tensorindex_list<ti::t, ti::a, ti::i>;

  // iat
  test_lhs_tensorindex_reorder_impl<iat_list, iat_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<iat_list, iat_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<iat_list, iat_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<iat_list, iat_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<iat_list, iat_list>::apply(symm_321);

  // ita
  test_lhs_tensorindex_reorder_impl<ita_list, iat_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<ita_list, ita_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<ita_list, iat_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<ita_list, ita_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<ita_list, ita_list>::apply(symm_321);

  // ait
  test_lhs_tensorindex_reorder_impl<ait_list, iat_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<ait_list, ait_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<ait_list, ait_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<ait_list, iat_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<ait_list, ait_list>::apply(symm_321);

  // ati
  test_lhs_tensorindex_reorder_impl<ati_list, iat_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<ati_list, ita_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<ati_list, ait_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<ati_list, ati_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<ati_list, ati_list>::apply(symm_321);

  // tia
  test_lhs_tensorindex_reorder_impl<tia_list, iat_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<tia_list, ait_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<tia_list, tia_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<tia_list, ita_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<tia_list, tia_list>::apply(symm_321);

  // tai
  test_lhs_tensorindex_reorder_impl<tai_list, iat_list>::apply(symm_111);
  test_lhs_tensorindex_reorder_impl<tai_list, iat_list>::apply(symm_121);
  test_lhs_tensorindex_reorder_impl<tai_list, tia_list>::apply(symm_211);
  test_lhs_tensorindex_reorder_impl<tai_list, ati_list>::apply(symm_221);
  test_lhs_tensorindex_reorder_impl<tai_list, tai_list>::apply(symm_321);
}

// Tests that the canonical ordering of a list of `TensorIndex`s done by
// `tenex::detail::get_reordered_tensorindex_values` for a rank 4 tensor
void test_lhs_tensorindex_reorder_rank4() {
  constexpr size_t num_indices = 4;
  const std::array<symm_datatype, num_indices> symm_1111{{1, 1, 1, 1}};
  const std::array<symm_datatype, num_indices> symm_1121{{1, 1, 2, 1}};
  const std::array<symm_datatype, num_indices> symm_1211{{1, 2, 1, 1}};
  const std::array<symm_datatype, num_indices> symm_2111{{2, 1, 1, 1}};
  const std::array<symm_datatype, num_indices> symm_1221{{1, 2, 2, 1}};
  const std::array<symm_datatype, num_indices> symm_2121{{2, 1, 2, 1}};
  const std::array<symm_datatype, num_indices> symm_2211{{2, 2, 1, 1}};
  const std::array<symm_datatype, num_indices> symm_2221{{2, 2, 2, 1}};
  const std::array<symm_datatype, num_indices> symm_2321{{2, 3, 2, 1}};
  const std::array<symm_datatype, num_indices> symm_3321{{3, 3, 2, 1}};
  const std::array<symm_datatype, num_indices> symm_4321{{4, 3, 2, 1}};

  using tbai_list = make_tensorindex_list<ti::t, ti::b, ti::a, ti::i>;
  using aitb_list = make_tensorindex_list<ti::a, ti::i, ti::t, ti::b>;
  using ibat_list = make_tensorindex_list<ti::i, ti::b, ti::a, ti::t>;
  using tiab_list = make_tensorindex_list<ti::t, ti::i, ti::a, ti::b>;
  using iabt_list = make_tensorindex_list<ti::i, ti::a, ti::b, ti::t>;
  using aitb_list = make_tensorindex_list<ti::a, ti::i, ti::t, ti::b>;
  using btia_list = make_tensorindex_list<ti::b, ti::t, ti::i, ti::a>;
  using abti_list = make_tensorindex_list<ti::a, ti::b, ti::t, ti::i>;
  using btai_list = make_tensorindex_list<ti::b, ti::t, ti::a, ti::i>;

  // tbai
  test_lhs_tensorindex_reorder_impl<tbai_list, iabt_list>::apply(symm_1111);
  test_lhs_tensorindex_reorder_impl<tbai_list, ibat_list>::apply(symm_1121);
  test_lhs_tensorindex_reorder_impl<tbai_list, ibat_list>::apply(symm_1211);
  test_lhs_tensorindex_reorder_impl<tbai_list, tiab_list>::apply(symm_2111);
  test_lhs_tensorindex_reorder_impl<tbai_list, iabt_list>::apply(symm_1221);
  test_lhs_tensorindex_reorder_impl<tbai_list, aitb_list>::apply(symm_2121);
  test_lhs_tensorindex_reorder_impl<tbai_list, btia_list>::apply(symm_2211);
  test_lhs_tensorindex_reorder_impl<tbai_list, abti_list>::apply(symm_2221);
  test_lhs_tensorindex_reorder_impl<tbai_list, abti_list>::apply(symm_2321);
  test_lhs_tensorindex_reorder_impl<tbai_list, btai_list>::apply(symm_3321);
  test_lhs_tensorindex_reorder_impl<tbai_list, tbai_list>::apply(symm_4321);

  using taji_list = make_tensorindex_list<ti::t, ti::a, ti::j, ti::i>;
  using iajt_list = make_tensorindex_list<ti::i, ti::a, ti::j, ti::t>;
  using tija_list = make_tensorindex_list<ti::t, ti::i, ti::j, ti::a>;
  using ijat_list = make_tensorindex_list<ti::i, ti::j, ti::a, ti::t>;
  using jita_list = make_tensorindex_list<ti::j, ti::i, ti::t, ti::a>;
  using atij_list = make_tensorindex_list<ti::a, ti::t, ti::i, ti::j>;
  using jati_list = make_tensorindex_list<ti::j, ti::a, ti::t, ti::i>;
  using atji_list = make_tensorindex_list<ti::a, ti::t, ti::j, ti::i>;
  using btai_list = make_tensorindex_list<ti::b, ti::t, ti::a, ti::i>;

  // taji
  test_lhs_tensorindex_reorder_impl<taji_list, ijat_list>::apply(symm_1111);
  test_lhs_tensorindex_reorder_impl<taji_list, iajt_list>::apply(symm_1121);
  test_lhs_tensorindex_reorder_impl<taji_list, iajt_list>::apply(symm_1211);
  test_lhs_tensorindex_reorder_impl<taji_list, tija_list>::apply(symm_2111);
  test_lhs_tensorindex_reorder_impl<taji_list, ijat_list>::apply(symm_1221);
  test_lhs_tensorindex_reorder_impl<taji_list, jita_list>::apply(symm_2121);
  test_lhs_tensorindex_reorder_impl<taji_list, atij_list>::apply(symm_2211);
  test_lhs_tensorindex_reorder_impl<taji_list, jati_list>::apply(symm_2221);
  test_lhs_tensorindex_reorder_impl<taji_list, jati_list>::apply(symm_2321);
  test_lhs_tensorindex_reorder_impl<taji_list, atji_list>::apply(symm_3321);
  test_lhs_tensorindex_reorder_impl<taji_list, taji_list>::apply(symm_4321);

  using akji_list = make_tensorindex_list<ti::a, ti::k, ti::j, ti::i>;
  using ijka_list = make_tensorindex_list<ti::i, ti::j, ti::k, ti::a>;
  using ikja_list = make_tensorindex_list<ti::i, ti::k, ti::j, ti::a>;
  using aijk_list = make_tensorindex_list<ti::a, ti::i, ti::j, ti::k>;
  using kaij_list = make_tensorindex_list<ti::k, ti::a, ti::i, ti::j>;
  using jiak_list = make_tensorindex_list<ti::j, ti::i, ti::a, ti::k>;
  using jkai_list = make_tensorindex_list<ti::j, ti::k, ti::a, ti::i>;
  using kaji_list = make_tensorindex_list<ti::k, ti::a, ti::j, ti::i>;

  // akji
  test_lhs_tensorindex_reorder_impl<akji_list, ijka_list>::apply(symm_1111);
  test_lhs_tensorindex_reorder_impl<akji_list, ikja_list>::apply(symm_1121);
  test_lhs_tensorindex_reorder_impl<akji_list, ikja_list>::apply(symm_1211);
  test_lhs_tensorindex_reorder_impl<akji_list, aijk_list>::apply(symm_2111);
  test_lhs_tensorindex_reorder_impl<akji_list, ijka_list>::apply(symm_1221);
  test_lhs_tensorindex_reorder_impl<akji_list, jiak_list>::apply(symm_2121);
  test_lhs_tensorindex_reorder_impl<akji_list, kaij_list>::apply(symm_2211);
  test_lhs_tensorindex_reorder_impl<akji_list, jkai_list>::apply(symm_2221);
  test_lhs_tensorindex_reorder_impl<akji_list, jkai_list>::apply(symm_2321);
  test_lhs_tensorindex_reorder_impl<akji_list, kaji_list>::apply(symm_3321);
  test_lhs_tensorindex_reorder_impl<akji_list, akji_list>::apply(symm_4321);

  using lkji_list = make_tensorindex_list<ti::l, ti::k, ti::j, ti::i>;
  using ijkl_list = make_tensorindex_list<ti::i, ti::j, ti::k, ti::l>;
  using ikjl_list = make_tensorindex_list<ti::i, ti::k, ti::j, ti::l>;
  using lijk_list = make_tensorindex_list<ti::l, ti::i, ti::j, ti::k>;
  using jilk_list = make_tensorindex_list<ti::j, ti::i, ti::l, ti::k>;
  using klij_list = make_tensorindex_list<ti::k, ti::l, ti::i, ti::j>;
  using jkli_list = make_tensorindex_list<ti::j, ti::k, ti::l, ti::i>;
  using klji_list = make_tensorindex_list<ti::k, ti::l, ti::j, ti::i>;

  // lkji
  test_lhs_tensorindex_reorder_impl<lkji_list, ijkl_list>::apply(symm_1111);
  test_lhs_tensorindex_reorder_impl<lkji_list, ikjl_list>::apply(symm_1121);
  test_lhs_tensorindex_reorder_impl<lkji_list, ikjl_list>::apply(symm_1211);
  test_lhs_tensorindex_reorder_impl<lkji_list, lijk_list>::apply(symm_2111);
  test_lhs_tensorindex_reorder_impl<lkji_list, ijkl_list>::apply(symm_1221);
  test_lhs_tensorindex_reorder_impl<lkji_list, jilk_list>::apply(symm_2121);
  test_lhs_tensorindex_reorder_impl<lkji_list, klij_list>::apply(symm_2211);
  test_lhs_tensorindex_reorder_impl<lkji_list, jkli_list>::apply(symm_2221);
  test_lhs_tensorindex_reorder_impl<lkji_list, jkli_list>::apply(symm_2321);
  test_lhs_tensorindex_reorder_impl<lkji_list, klji_list>::apply(symm_3321);
  test_lhs_tensorindex_reorder_impl<lkji_list, lkji_list>::apply(symm_4321);
}

// Tests `tenex::detail::get_reordered_tensorindex_values`
void test_lhs_tensorindex_reorder() {
  test_lhs_tensorindex_reorder_symm_consistency();
  test_lhs_tensorindex_reorder_rank0();
  test_lhs_tensorindex_reorder_rank1();
  test_lhs_tensorindex_reorder_rank2();
  test_lhs_tensorindex_reorder_rank3();
  test_lhs_tensorindex_reorder_rank4();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Evaluate",
                  "[DataStructures][Unit]") {
  test_contains_indices_to_contract();
  test_lhs_tensorindex_reorder();
  test_evaluate_and_canon_multi_index_consistency();
}
