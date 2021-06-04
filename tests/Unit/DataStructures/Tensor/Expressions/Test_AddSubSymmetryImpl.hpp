// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace detail {
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

  return outsymm;
}

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

  using type = tmpl::integral_list<std::int32_t, outsymm[Ints]...>;
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
