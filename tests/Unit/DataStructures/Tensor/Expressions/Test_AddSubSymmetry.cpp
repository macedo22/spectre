// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

template <typename... T>
struct td;

template <size_t I>
struct TensorIndex {
  static constexpr size_t value = I;
  static constexpr bool is_spacetime = I < 10;
};

using ti_a = TensorIndex<0>;
using ti_b = TensorIndex<1>;
using ti_c = TensorIndex<2>;
using ti_d = TensorIndex<3>;
using ti_e = TensorIndex<4>;

using ti_i = TensorIndex<10>;
using ti_j = TensorIndex<11>;
using ti_k = TensorIndex<12>;
using ti_l = TensorIndex<13>;
using ti_m = TensorIndex<14>;

namespace detail {
template <typename State, typename SymmValue, typename Iteration,
          typename SymmValueToFind>
struct PositionsOfSymmValue {
  static_assert(SymmValueToFind::value == 1 or SymmValueToFind::value == 2,
                "oops SymmValueToFind");
  static_assert(SymmValue::value == 1 or SymmValue::value == 2,
                "oops SymmValue");
  // static_assert(std::is_same_v<Iteration, tmpl::int32_t<0>> or
  // std::is_same_v<Iteration, tmpl::int32_t<1>>,"oops Iteration");
  using type =
      typename std::conditional_t<SymmValue::value == SymmValueToFind::value,
                                  tmpl::push_back<State, Iteration>, State>;
};

template <typename State, typename SymmValueToFind, typename Symm/*,
          typename SymmValueToFind*/>
struct AddSubtractSymmetryImpl {
  using positions_of_symm_value = tmpl::enumerated_fold<
      Symm, tmpl::list<>,
      PositionsOfSymmValue<tmpl::_state, tmpl::_element, tmpl::_3,
                           tmpl::pin<SymmValueToFind>>,
      tmpl::int32_t<0>>;
  // static_assert(std::is_same_v<positions_of_symm_value, double>,"oops
  // positions_of_symm_value"); static_assert(std::is_same_v<SymmValueToFind,
  // double>,"oops SymmValueToFind");
  // static_assert(std::is_same_v<tmpl::append<SymmValueToFind,
  // positions_of_symm_value>, double>,"oops append");
  using type = typename std::conditional_t<
      (tmpl::size<positions_of_symm_value>::value > 0),
      tmpl::push_back<
          State, tmpl::push_front<positions_of_symm_value, SymmValueToFind>>,
      State>;
  // static_assert(std::is_same_v<type, double>,"oops type");
};

// template <typename State, typename Element, typename Symm/*,
//           typename SymmValuesSet*/>
// struct AddSubtractSymmetryHelper {
//   using symmetric_index_positions = typename tmpl::fold<
//       SymmValuesSet, tmpl::list<>,
//       AddSubtractSymmetryImpl<tmpl::_state, tmpl::_element,
//       tmpl::pin<Symm>>>;
//   using type = symmetric_index_positions;
// };

template <typename SymmList1, typename SymmList2, typename TensorIndexList1,
          typename TensorIndexList2,
          size_t NumIndices = tmpl::size<SymmList1>::value,
          typename IndexSequence = std::make_index_sequence<NumIndices>>
struct AddSubtractSymmetry;

// template <template <typename...> class SymmList1, typename... Symm1,
//           template <typename...> class SymmList2, typename... Symm2,
//           template <typename...> class TensorIndexList1,
//           typename... TensorIndices1,
//           template <typename...> class TensorIndexList2,
//           typename... TensorIndices2, size_t NumIndices, size_t... Ints>
// struct AddSubtractSymmetry<SymmList1<Symm1...>, SymmList2<Symm2...>,
//                            TensorIndexList1<TensorIndices1...>,
//                            TensorIndexList2<TensorIndices2...>, NumIndices,
//                            std::index_sequence<Ints...>> {
//   static constexpr std::array<std::int32_t, NumIndices> symm1 = {
//       {Symm1::value...}};
//   static constexpr std::array<std::int32_t, NumIndices> symm2 = {
//       {Symm2::value...}};
//   static constexpr std::int32_t max_symm1 = *alg::max_element(symm1);
//   static constexpr std::int32_t min_symm1 = *alg::min_element(symm1);
//   // static_assert(1 == min_symm1,"oops min");
//   // static_assert(2 == max_symm1,"oops max");
//   // static_assert(std::is_same_v<tmpl::range<std::int32_t, min_symm1,
//   max_symm1
//   // + 1>, tmpl::integral_list<std::int32_t, 1, 2>>,"oops range");

//   using symmetric_index_positions = tmpl::fold<tmpl::range<std::int32_t,
//                                                            min_symm1,
//                                                            max_symm1 + 1>,
//                                                tmpl::list<>,
//                                                AddSubtractSymmetryImpl<
//                                                    tmpl::_state,
//                                                    tmpl::_element, tmpl::
//                                                        pin<
//                                                            SymmList1<
//                                                                Symm1...>> /*,
//               tmpl::pin<tmpl::range<std::int32_t, min_symm1, max_symm1 +
//               1>>*/>>;

//   using type = symmetric_index_positions;  // remove, just for testing for
//   now

//   //   static constexpr auto
// };

template <size_t NumIndices, Requires<(NumIndices < 2)> = nullptr>
constexpr std::array<std::int32_t, NumIndices> get_outsymm(
    const std::array<std::int32_t, NumIndices>& symm1,
    const std::array<std::int32_t, NumIndices>& /*symm2*/) {
  return symm1;
}

template <size_t NumIndices, Requires<(NumIndices >= 2)> = nullptr>
constexpr std::array<std::int32_t, NumIndices> get_outsymm(
    const std::array<std::int32_t, NumIndices>& symm1,
    const std::array<std::int32_t, NumIndices>& symm2) {
  std::array<std::int32_t, NumIndices> outsymm{};
  size_t right_index = NumIndices - 1;
  std::int32_t symm_value_to_set = 1;

  // loop over symm1 from right to left
  while (right_index < NumIndices) {
    std::int32_t symm1_value_to_find = symm1[right_index];
    std::int32_t symm2_value_to_find = symm2[right_index];
    if (outsymm[right_index] == 0) {
      outsymm[right_index] = symm_value_to_set;
      // loop over symm1 and symm2 looking for overlap
      for (size_t left_index = right_index - 1; left_index < NumIndices;
           left_index--) {
        // if we haven't yet set this index and there is an overlap
        if (outsymm[left_index] == 0 and
            symm1[left_index] == symm1_value_to_find and
            symm2[left_index] == symm2_value_to_find) {
          outsymm[left_index] = symm_value_to_set;
        }
      }
      symm_value_to_set++;
    }
    right_index--;
  }

  // // loop over all symm
  // while (symm1_value_to_find <= max_symm1) {
  //   bool backtrack = true;
  //   size_t index = NumIndices - 1;
  //   size_t backtrack_index = index;//NumIndices - 1; // ?
  //   while (backtrack) {
  //     backtrack = false;

  //     outsymm[backtrack_index] = symm1_value_to_insert;

  //     //size_t backtrack_index = NumIndices - 1;
  //     std::int32_t symm2_value_to_find = symm2[backtrack_index];

  //     for (size_t left_symm1_index = 0; left_symm1_index < index;
  //     left_symm1_index++) {
  //       // if overlap
  //       if (symm1[left_symm1_index] == symm1_value_to_find and
  //       symm2[left_symm1_index] == symm2_value_to_find) {
  //         outsymm[left_symm1_index] = symm1_value_to_insert;
  //       } else if (symm1[left_symm1_index] == symm1_value_to_find) { // if no
  //       overlap
  //         backtrack = true;
  //         backtrack_index = left_symm1_index;
  //       } // else, symm1 value not a match anyway
  //     }
  //     index = backtrack_index;
  //     symm1_value_to_insert++;
  //   }
  //   symm1_value_to_find++;
  // }

  // close to working:
  // // loop over all symm
  // while (symm1_value_to_find <= max_symm1) {
  //   bool backtrack = true;
  //   size_t backtrack_index = NumIndices - 1; // ?
  //   while (backtrack) {
  //     backtrack = false;

  //     //size_t backtrack_index = NumIndices - 1;
  //     std::int32_t symm2_value_to_find = symm2[backtrack_index];

  //     for (size_t left_symm1_index = 0; left_symm1_index < backtrack_index;
  //     left_symm1_index++) {
  //       // if overlap
  //       if (symm1[left_symm1_index] == symm1_value_to_find and
  //       symm2[left_symm1_index] == symm2_value_to_find) {
  //         outsymm[left_symm1_index] = symm1_value_to_find;

  //       } else if (symm1[left_symm1_index] == symm1_value_to_find) { // if no
  //       overlap
  //         backtrack = true;
  //         backtrack_index = left_symm1_index;
  //       } // else, symm1 value not a match anyway
  //     }
  //     symm1_value_to_insert++;
  //   }
  //   symm1_value_to_find++;
  // }

  // mostly worked:
  //
  // std::array<std::int32_t, NumIndices> outsymm = symm1; // same as symm1 to
  // start std::int32_t symm_value_to_find = min_symm1; size_t symm_index =
  // NumIndices - 1; std::int32_t num_replaced = 0; while (symm_value_to_find <=
  // max_symm1 and symm_index < NumIndices) {
  //   const std::int32_t current_symm_value = symm1[symm_index];
  //   if (current_symm_value == symm_value_to_find) {
  //     const std::int32_t current_other_symm_value = symm2[symm_index];
  //     for (size_t j = symm_index - 1; j < NumIndices; j--) {
  //       const std::int32_t compare_current_symm_value = symm1[j];
  //       const std::int32_t compare_current_other_symm_value = symm2[j];
  //       if (current_symm_value == compare_current_symm_value and
  //       current_other_symm_value != compare_current_other_symm_value) {
  //         outsymm[j] += max_symm1 + num_replaced;
  //         num_replaced++;
  //       }
  //     }
  //     symm_value_to_find++;
  //   }
  //   symm_index--;
  // }

  return outsymm;
}

template <template <typename...> class SymmList1, typename... Symm1,
          template <typename...> class SymmList2, typename... Symm2,
          template <typename...> class TensorIndexList1,
          typename... TensorIndices1,
          template <typename...> class TensorIndexList2,
          typename... TensorIndices2, size_t NumIndices, size_t... Ints>
struct AddSubtractSymmetry<SymmList1<Symm1...>, SymmList2<Symm2...>,
                           TensorIndexList1<TensorIndices1...>,
                           TensorIndexList2<TensorIndices2...>, NumIndices,
                           std::index_sequence<Ints...>> {
  static constexpr std::array<size_t, NumIndices> lhs_tensorindex_values = {
      {TensorIndices1::value...}};
  static constexpr std::array<size_t, NumIndices> rhs_tensorindex_values = {
      {TensorIndices2::value...}};
  static constexpr std::array<size_t, NumIndices> lhs_to_rhs_map = {
      {std::distance(
          rhs_tensorindex_values.begin(),
          alg::find(rhs_tensorindex_values, lhs_tensorindex_values[Ints]))...}};

  static constexpr std::array<std::int32_t, NumIndices> symm1 = {
      {Symm1::value...}};
  static constexpr std::array<std::int32_t, NumIndices> symm2 = {
      {Symm2::value...}};
  static constexpr std::array<std::int32_t, NumIndices> outsymm =
      get_outsymm(symm1, {{symm2[lhs_to_rhs_map[Ints]]...}});

  using type = Symmetry<outsymm[Ints]...>;  // temporary

  //   static constexpr std::int32_t max_symm1 = *alg::max_element(symm1);
  //   static constexpr std::int32_t min_symm1 = *alg::min_element(symm1);
  //   static_assert(min_symm1 == symm1[NumIndices - 1], "Algorithm assumes last
  //   element of symm1 is the smallest value.");

  //   std::array<std::int32_t, NumIndices> outsymm = {
  //       {Symm1::value...}}; // same as symm1 to start
  //   std::int32_t symm_value_to_find = min_symm1;
  //   size_t symm_index = NumIndices - 1;
  //   std::int32_t num_replaced = 0;
  //   while (symm_value_to_find <= max_symm1 and symm_index >= 0) {
  //     const std::int32_t current_symm_value = symm1[symm_index];
  //     if (current_symm_value == symm_value_to_find) {
  //       const std::int32_t current_other_symm_value = symm2[symm_index];
  //       for (size_t j = i - 1; j >= 0; j--) {
  //         const std::int32_t compare_current_symm_value = symm1[j];
  //         const std::int32_t compare_current_other_symm_value = symm2[j];
  //         if (current_symm_value == compare_current_symm_value and
  //         current_other_symm_value == compare_current_other_symm_value) {
  //           outsymm[j] += max_symm1 + num_replaced;
  //           num_replaced++;
  //         }
  //       }
  //       symm_value_to_find++;
  //     }
  //     symm_index--;
  //   }

  //   std::int32_t symm_value_to_find = min_symm1;
  //   size_t symm_index = NumIndices - 1;
  //   while (symm_value_to_find >= 0) {
  //     for (size_t symm_index = NumIndices - 1; symm_index >= 0; symm_index++)
  //     {
  //       const std::int32_t current_symm_value = symm1[symm_index];
  //       if (current_symm_value == symm_value_to_find) {
  //         for (size_t symm_index2 = symm_index - 1; symm_index2 >= 0;
  //         symm_index2++) {

  //         }
  //       }
  //     }
  //     symm_index--;
  //   }
  //   static constexpr auto
};

template <typename T1, typename T2, typename SymmList1 = typename T1::symmetry,
          typename SymmList2 = typename T2::symmetry,
          typename TensorIndexList1 = typename T1::args_list,
          typename TensorIndexList2 = typename T2::args_list>
struct AddSubtractType;

template <typename T1, typename T2, template <typename...> class SymmList1,
          typename... Symm1, template <typename...> class SymmList2,
          typename... Symm2, template <typename...> class TensorIndexList1,
          typename... TensorIndices1,
          template <typename...> class TensorIndexList2,
          typename... TensorIndices2>
struct AddSubtractType<T1, T2, SymmList1<Symm1...>, SymmList2<Symm2...>,
                       TensorIndexList1<TensorIndices1...>,
                       TensorIndexList2<TensorIndices2...>> {
  using type =
      std::conditional_t<std::is_same<typename T1::type, DataVector>::value or
                             std::is_same<typename T2::type, DataVector>::value,
                         DataVector, double>;
  using symmetry =
      typename AddSubtractSymmetry<SymmList1<Symm1...>, SymmList2<Symm2...>,
                                   TensorIndexList1<TensorIndices1...>,
                                   TensorIndexList2<TensorIndices2...>>::type;
  using index_list = tmpl::append<typename T1::index_list>;
  using tensorindex_list = tmpl::append<typename T1::args_list>;
};
}  // namespace detail

template <size_t NumIndices>
std::array<std::int32_t, NumIndices> get_outsymm(
    const std::array<std::int32_t, NumIndices>& symm1,
    const std::array<std::int32_t, NumIndices>& symm2) {
  const std::int32_t max_symm1 = *alg::max_element(symm1);
  const std::int32_t min_symm1 = *alg::min_element(symm1);
  static_assert(NumIndices > 1, "oops");
  // assert(min_symm1 == symm1[NumIndices - 1], "oops");

  std::array<std::int32_t, NumIndices>
      outsymm{};  // = symm1; // same as symm1 to start

  // size_t index = NumIndices - 1;
  // std::int32_t symm1_value_to_find = min_symm1;//symm1[NumIndices - 1];
  std::int32_t symm1_value_to_insert = min_symm1;

  size_t outer_right_index = NumIndices - 1;

  // loop over all symm and symm indices
  while (outer_right_index < NumIndices) {
    std::int32_t symm1_value_to_find = symm1[outer_right_index];
    std::int32_t symm2_value_to_find = symm2[outer_right_index];

    size_t inner_right_index = outer_right_index;
    // size_t inner_left_index = inner_right_index - 1;
    if (outsymm[inner_right_index] == 0) {
      // keep iterating while
      outsymm[inner_right_index] = symm1_value_to_insert;
      for (size_t inner_left_index = inner_right_index - 1;
           inner_left_index < NumIndices; inner_left_index--) {
        // if we haven't yet set this index and there is an overlap
        if (outsymm[inner_left_index] == 0 and
            symm1[inner_left_index] == symm1_value_to_find and
            symm2[inner_left_index] == symm2_value_to_find) {
          outsymm[inner_left_index] = symm1_value_to_insert;
        }
      }
      symm1_value_to_insert++;
    }
    outer_right_index--;
  }

  return outsymm;
}

template <typename Symm1, typename Symm2>
using current_addsubtract_symmetry =
    tmpl::transform<Symm1, Symm2, tmpl::append<tmpl::max<tmpl::_1, tmpl::_2>>>;

void test1() {
  using symm1 = Symmetry<2, 1, 1, 1, 1>;
  using symm2 = Symmetry<3, 2, 2, 1, 1>;
  // using symm2 = Symmetry<2, 2, 2, 1, 1>;
  // using symm2 = Symmetry<1, 2, 2, 1, 1>;

  using tensorindex_list1 = tmpl::list<ti_a, ti_b, ti_c, ti_d, ti_e>;
  using tensorindex_list2 = tmpl::list<ti_a, ti_b, ti_c, ti_d, ti_e>;

  using new_result_symmetry =
      typename detail::AddSubtractSymmetry<symm1, symm2, tensorindex_list1,
                                           tensorindex_list2>::type;

  static_assert(
      std::is_same_v<new_result_symmetry,
                     tmpl::integral_list<std::int32_t, 3, 2, 2, 1, 1>>,
      "oops");

  // using tensorindex_list1 = tmpl::list<ti_a, ti_b>;
  // using tensorindex_list2 = tmpl::list<ti_a, ti_b>;

  // auto outsymm = get_outsymm<5>({{2, 1, 1, 1, 1}}, {{3, 2, 2, 1, 1}});
  // std::cout << "outsymm : [" << outsymm[0];
  // for (size_t i = 1; i < outsymm.size(); i++) {
  //   std::cout << ", " << outsymm[i];
  // }
  // std::cout << "]" << std::endl;

  // using current_result_symmetry = current_addsubtract_symmetry<symm1, symm2>;

  // std::cout << "current_result_symmetry : ";
  // std::cout << "["
  //           << static_cast<std::int32_t>(
  //                  tmpl::at_c<current_result_symmetry, 0>::value)
  //           << ", "
  //           << static_cast<std::int32_t>(
  //                  tmpl::at_c<current_result_symmetry, 1>::value)
  //           << ", "
  //           << static_cast<std::int32_t>(
  //                  tmpl::at_c<current_result_symmetry, 2>::value)
  //           << ", "
  //           << static_cast<std::int32_t>(
  //                  tmpl::at_c<current_result_symmetry, 3>::value)
  //           << ", "
  //           << static_cast<std::int32_t>(
  //                  tmpl::at_c<current_result_symmetry, 4>::value)
  //           << "]" << std::endl;

  // using new_result_symmetry =
  //     typename detail::AddSubtractSymmetry<symm1, symm2,
  //                                          tensorindex_list1,
  //                                          tensorindex_list2>::type;

  // std::cout << "new_result_symmetry : ";
  // std::cout << "["
  //           << static_cast<std::int32_t>(
  //                  tmpl::at_c<new_result_symmetry, 0>::value)
  //           << ", "
  //           << static_cast<std::int32_t>(
  //                  tmpl::at_c<new_result_symmetry, 1>::value)
  //           << ", "
  //           << static_cast<std::int32_t>(
  //                  tmpl::at_c<new_result_symmetry, 2>::value)
  //           << ", "
  //           << static_cast<std::int32_t>(
  //                  tmpl::at_c<new_result_symmetry, 3>::value)
  //           << ", "
  //           << static_cast<std::int32_t>(
  //                  tmpl::at_c<new_result_symmetry, 4>::value)
  //           << "]" << std::endl;

  // td<symmetric_index_positions> idk;

  //   using new_result_symmetry = detail::AddSubtractSymmetry<symm1, symm2,
  //   tensorindex_list1, tensorindex_list2>;

  //   std::cout << "new_result_symmetry : ";
  //   std::cout << "[" <<
  //   static_cast<std::int32_t>(tmpl::at_c<new_result_symmetry, 0>::value) <<
  //   ", "
  //       << static_cast<std::int32_t>(tmpl::at_c<new_result_symmetry,
  //       1>::value) << ", "
  //       << static_cast<std::int32_t>(tmpl::at_c<new_result_symmetry,
  //       2>::value) << ", "
  //       << static_cast<std::int32_t>(tmpl::at_c<new_result_symmetry,
  //       3>::value) << ", "
  //       << static_cast<std::int32_t>(tmpl::at_c<new_result_symmetry,
  //       4>::value) << "]" << std::endl;
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubSymmetry",
                  "[DataStructures][Unit]") {
  test1();
}
