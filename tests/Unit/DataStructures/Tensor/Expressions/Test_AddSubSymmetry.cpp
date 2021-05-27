// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
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
  static constexpr std::array<std::int32_t, NumIndices> symm1 = {
      {Symm1::value...}};
  static constexpr std::array<std::int32_t, NumIndices> symm2 = {
      {Symm2::value...}};
  static constexpr std::int32_t max_symm1 = *alg::max_element(symm1);
  static constexpr std::int32_t min_symm1 = *alg::min_element(symm1);
  // static_assert(1 == min_symm1,"oops min");
  // static_assert(2 == max_symm1,"oops max");
  // static_assert(std::is_same_v<tmpl::range<std::int32_t, min_symm1, max_symm1
  // + 1>, tmpl::integral_list<std::int32_t, 1, 2>>,"oops range");

  using symmetric_index_positions = tmpl::fold<tmpl::range<std::int32_t,
                                                           min_symm1,
                                                           max_symm1 + 1>,
                                               tmpl::list<>,
                                               AddSubtractSymmetryImpl<
                                                   tmpl::_state, tmpl::_element,
                                                   tmpl::
                                                       pin<
                                                           SymmList1<
                                                               Symm1...>> /*,
              tmpl::pin<tmpl::range<std::int32_t, min_symm1, max_symm1 +
              1>>*/>>;

  using type = symmetric_index_positions;  // remove, just for testing for now

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
  using symmetry = AddSubtractSymmetry<SymmList1<Symm1...>, SymmList2<Symm2...>,
                                       TensorIndexList1<TensorIndices1...>,
                                       TensorIndexList2<TensorIndices2...>>;
  using index_list = tmpl::append<typename T1::index_list>;
  using tensorindex_list = tmpl::append<typename T1::args_list>;
};
}  // namespace detail

template <typename Symm1, typename Symm2>
using current_addsubtract_symmetry =
    tmpl::transform<Symm1, Symm2, tmpl::append<tmpl::max<tmpl::_1, tmpl::_2>>>;

void test1() {
  using symm1 =
      Symmetry<2, 1, 1, 1, 2>;  // -> <3, 3, 3, 3, 2> -> <6, 5, 4, 3, 2>
  // td<symm1> s;
  using symm2 = Symmetry<4, 3, 2, 1, 1>;

  using current_result_symmetry = current_addsubtract_symmetry<symm1, symm2>;

  std::cout << "current_result_symmetry : ";
  std::cout << "["
            << static_cast<std::int32_t>(
                   tmpl::at_c<current_result_symmetry, 0>::value)
            << ", "
            << static_cast<std::int32_t>(
                   tmpl::at_c<current_result_symmetry, 1>::value)
            << ", "
            << static_cast<std::int32_t>(
                   tmpl::at_c<current_result_symmetry, 2>::value)
            << ", "
            << static_cast<std::int32_t>(
                   tmpl::at_c<current_result_symmetry, 3>::value)
            << ", "
            << static_cast<std::int32_t>(
                   tmpl::at_c<current_result_symmetry, 4>::value)
            << "]" << std::endl;

  using tensorindex_list1 = tmpl::list<ti_a, ti_b, ti_c, ti_d, ti_e>;
  using tensorindex_list2 = tmpl::list<ti_a, ti_b, ti_c, ti_d, ti_e>;

  using symmetric_index_positions =
      typename detail::AddSubtractSymmetry<Symmetry<2, 1>, Symmetry<1, 1>,
                                           tensorindex_list1,
                                           tensorindex_list2>::type;
  td<symmetric_index_positions> idk;

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
