// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Expression Templates for contracting tensor indices on a single
/// Tensor

#pragma once

#include <array>
#include <cstddef>
// #include <iostream>
#include <limits>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup TensorExpressionsGroup
 * Holds all possible TensorExpressions currently implemented
 */
namespace TensorExpressions {

namespace detail {

template <typename I1, typename I2>
using indices_contractible = std::bool_constant<
    I1::ul != I2::ul and
    std::is_same_v<typename I1::Frame, typename I2::Frame> and
    ((I1::index_type == I2::index_type and I1::dim == I2::dim) or
     // If one index is spacetime and the other is spatial, the indices can
     // be contracted if they have the same number of spatial dimensions
     (I1::index_type == IndexType::Spacetime and I1::dim == I2::dim + 1) or
     (I2::index_type == IndexType::Spacetime and I1::dim + 1 == I2::dim))>;

// template <typename UncontractedIndexList, size_t NumIndexPairsToContract,
//           typename IndexPairsIndexSequence =
//               std::make_index_sequence<NumIndexPairsToContract>>
// struct indices_contractible;

// template <typename... UncontractedIndices, size_t NumIndexPairsToContract,
//           size_t... IndexPairsInts>
// struct indices_contractible<tmpl::list<UncontractedIndices...>,
//                             NumIndexPairsToContract,
//                             std::index_sequence<IndexPairsInts...>> {
//   static constexpr bool apply(
//       const std::array<std::pair<size_t, size_t>, NumIndexPairsToContract>&
//           contracted_index_map) {
//     return (
//         (... and
//          (indices_contractible_impl<
//              typename tmpl::at_c<tmpl::list<UncontractedIndices...>,
//                                  contracted_index_map[IndexPairsInts].first>,
//              typename tmpl::at_c<tmpl::list<UncontractedIndices...>,
//                               contracted_index_map[IndexPairsInts].second>>::
//               value)));
//   }
// };

// template <size_t NumIndexPairsToContract>
// constexpr bool indices_contractible(const std::array<std::pair<size_t,
// size_t>, NumIndexPairsToContract>&
//       contracted_index_map, std::index_sequence<I...>) {

// }

template <size_t NumUncontractedIndices>
constexpr size_t get_num_contracted_index_pairs(
    const std::array<size_t, NumUncontractedIndices>&
        uncontracted_tensor_index_values) {
  size_t count = 0;
  for (size_t i = 0; i < NumUncontractedIndices; i++) {
    const size_t current_value = gsl::at(uncontracted_tensor_index_values, i);
    // Concrete time indices are not contracted
    if (not detail::is_time_index_value(current_value)) {
      const size_t opposite_value_to_find =
          get_tensorindex_value_with_opposite_valence(current_value);
      for (size_t j = i + 1; j < NumUncontractedIndices; j++) {
        if (opposite_value_to_find ==
            gsl::at(uncontracted_tensor_index_values, j)) {
          // We found both the lower and upper version of a generic index in the
          // list of generic indices, so we return this pair's positions
          count++;
        }
      }
    }
  }

  return count;
}

// TODO : add a Requires<(NumUncontractedIndices >= 2)>
template <size_t NumContractedIndexPairs, size_t NumUncontractedIndices>
constexpr std::pair<
    std::array<size_t, NumUncontractedIndices - NumContractedIndexPairs * 2>,
    std::array<std::pair<size_t, size_t>, NumContractedIndexPairs>>
get_index_maps(
    const std::array<size_t, NumUncontractedIndices>& tensor_index_values) {
  // constexpr size_t num_indices_not_contracted =
  //     NumUncontractedIndices - NumContractedIndexPairs * 2;
  std::array<std::pair<size_t, size_t>, NumContractedIndexPairs>
      contracted_index_map{};
  // static_assert(NumContractedIndexPairs == 1);
  // need to initialize, or gsl::at complains
  // for (auto& pair : contracted_index_map) {
  //   pair.first = std::numeric_limits<double>::signaling_NaN();
  //   pair.second = std::numeric_limits<double>::signaling_NaN();
  // }
  // static_assert(NumUncontractedIndices == 4);
  std::array<size_t, NumUncontractedIndices - NumContractedIndexPairs * 2>
      not_contracted_index_map{};

  std::array<bool, NumUncontractedIndices> index_mapping_set{};
  for (size_t i = 0; i < NumUncontractedIndices; i++) {
    gsl::at(index_mapping_set, i) = false;
  }

  size_t contracted_map_index_to_assign = 0;
  size_t not_contracted_map_index_to_assign =
      NumUncontractedIndices - NumContractedIndexPairs * 2 - 1;
  for (size_t i = NumUncontractedIndices - 1; i < NumUncontractedIndices; i--) {
    if (not gsl::at(index_mapping_set, i)) {
      const size_t current_value = gsl::at(tensor_index_values, i);
      // Concrete time indices are not contracted
      if (not detail::is_time_index_value(current_value)) {
        const size_t opposite_value_to_find =
            get_tensorindex_value_with_opposite_valence(current_value);
        for (size_t j = i - 1; j < NumUncontractedIndices; j--) {
          if (opposite_value_to_find == gsl::at(tensor_index_values, j)) {
            // We found both the lower and upper version of a generic index in
            // the list of generic indices, so we return this pair's positions
            gsl::at(contracted_index_map, contracted_map_index_to_assign)
                .first = j;
            gsl::at(contracted_index_map, contracted_map_index_to_assign)
                .second = i;
            contracted_map_index_to_assign++;
            gsl::at(index_mapping_set, j) = true;
            gsl::at(index_mapping_set, i) = true;
            break;
          }
        }
      }
      if (not gsl::at(index_mapping_set, i)) {
        gsl::at(not_contracted_index_map, not_contracted_map_index_to_assign) =
            i;
        not_contracted_map_index_to_assign--;
      }
    }
  }

  return std::pair{not_contracted_index_map, contracted_index_map};
}

template <typename UncontractedTensorExpression, typename DataType,
          typename UncontractedSymm, typename UncontractedIndexList,
          typename UncontractedTensorIndexList, size_t NumContractedIndices,
          size_t NumIndexPairsToContract,
          typename ContractedIndexSequence =
              std::make_index_sequence<NumContractedIndices>,
          typename IndexPairsToContractSequence =
              std::make_index_sequence<NumIndexPairsToContract>>
struct ContractedType;

template <typename UncontractedTensorExpression, typename DataType,
          template <typename...> class UncontractedSymmList,
          typename... UncontractedSymm,
          template <typename...> class UncontractedIndexList,
          typename... UncontractedIndices,
          template <typename...> class UncontractedTensorIndexList,
          typename... UncontractedTensorIndices, size_t NumContractedIndices,
          size_t NumIndexPairsToContract, size_t... ContractedInts,
          size_t... IndexPairsToContractInts>
struct ContractedType<UncontractedTensorExpression, DataType,
                      UncontractedSymmList<UncontractedSymm...>,
                      UncontractedIndexList<UncontractedIndices...>,
                      UncontractedTensorIndexList<UncontractedTensorIndices...>,
                      NumContractedIndices, NumIndexPairsToContract,
                      std::index_sequence<ContractedInts...>,
                      std::index_sequence<IndexPairsToContractInts...>> {
  // static_assert(NumContractedIndices == 0);
  static constexpr size_t num_uncontracted_tensor_indices =
      sizeof...(UncontractedTensorIndices);
  // static_assert(num_uncontracted_tensor_indices == 2);
  static constexpr std::array<size_t, num_uncontracted_tensor_indices>
      uncontracted_tensorindex_values = {{UncontractedTensorIndices::value...}};
  // TODO : call this in contract() so that TensorContract is called with # of
  // index pairs to contract
  //   static constexpr size_t num_contracted_index_pairs =
  //       get_num_contracted_index_pairs<num_uncontracted_tensor_indices>(
  //           uncontracted_tensorindex_values);
  static constexpr size_t num_indices_to_contract =
      num_uncontracted_tensor_indices - NumContractedIndices;
  // static_assert(num_indices_to_contract == 2);
  static constexpr size_t num_contracted_index_pairs =
      num_indices_to_contract / 2;
  // static_assert(num_contracted_index_pairs == 1);
  //   static constexpr size_t num_contracted_indices =
  //       num_uncontracted_tensor_indices - num_indices_to_contract;
  static constexpr inline std::pair<
      std::array<size_t, NumContractedIndices>,
      std::array<std::pair<size_t, size_t>, num_contracted_index_pairs>>
      index_maps = get_index_maps<num_contracted_index_pairs,
                                  num_uncontracted_tensor_indices>(
          uncontracted_tensorindex_values);

  // static_assert(
  //     indices_contractible<UncontractedIndexList<UncontractedIndices...>,
  //                          num_contracted_index_pairs>::apply(index_maps
  //                                                                 .second),
  //     "Cannot contract the requested indices.");//IndexPairsToContractInts
  static_assert(
      ((... and
        (indices_contractible<
            typename tmpl::at_c<
                tmpl::list<UncontractedIndices...>,
                index_maps.second[IndexPairsToContractInts].first>,
            typename tmpl::at_c<
                tmpl::list<UncontractedIndices...>,
                index_maps.second[IndexPairsToContractInts].second>>::value))),
      "Cannot contract the requested indices.");

  //   static constexpr inline std::array<size_t,
  //   num_uncontracted_tensor_indices>
  //       uncontracted_index_dims = {{Indices::dim...}};
  //   static constexpr inline std::array<size_t,
  //   num_uncontracted_tensor_indices>
  //       uncontracted_index_types = {{Indices::index_type...}};

  static constexpr inline std::array<IndexType, num_uncontracted_tensor_indices>
      uncontracted_index_types = {{UncontractedIndices::index_type...}};

  static constexpr inline std::array<std::pair<size_t, size_t>,
                                     num_contracted_index_pairs>
      contracted_index_shifts = []() {
        std::array<std::pair<size_t, size_t>, num_contracted_index_pairs>
            shifts{};
        for (size_t i = 0; i < num_contracted_index_pairs; i++) {
          gsl::at(shifts, i).first = static_cast<size_t>(
              gsl::at(uncontracted_index_types,
                      gsl::at(index_maps.second, i).first) ==
                  IndexType::Spacetime and
              gsl::at(uncontracted_tensorindex_values,
                      gsl::at(index_maps.second, i).first) >=
                  TensorIndex_detail::spatial_sentinel);
          gsl::at(shifts, i).second = static_cast<size_t>(
              gsl::at(uncontracted_index_types,
                      gsl::at(index_maps.second, i).second) ==
                  IndexType::Spacetime and
              gsl::at(uncontracted_tensorindex_values,
                      gsl::at(index_maps.second, i).second) >=
                  TensorIndex_detail::spatial_sentinel);
        }
        return shifts;
      }();

  static constexpr inline std::array<size_t, num_uncontracted_tensor_indices>
      uncontracted_index_dims = {{UncontractedIndices::dim...}};

  // TODO : add a static_assert that makes sure this is positive?
  static constexpr size_t num_terms_summed =  // 2 * 3 * 4;
      []() {
        size_t num_terms = gsl::at(uncontracted_index_dims,
                                   gsl::at(index_maps.second, 0).first) -
                           //  gsl::at(contracted_index_shifts,
                           //          gsl::at(index_maps.second, 0).first);
                           gsl::at(contracted_index_shifts, 0).first;
        for (size_t i = 1; i < num_contracted_index_pairs; i++) {
          num_terms *= gsl::at(uncontracted_index_dims,
                               gsl::at(index_maps.second, i).first) -
                       //  gsl::at(contracted_index_shifts,
                       //          gsl::at(index_maps.second, i).first);
                       gsl::at(contracted_index_shifts, i).first;
        }
        return num_terms;
      }();
  static_assert(num_terms_summed > 0,
                "There should be a non-zero number of components to sum in the "
                "contraction.");

  using symmetry =
      Symmetry<tmpl::at_c<UncontractedSymmList<UncontractedSymm...>,
                          index_maps.first[ContractedInts]>::value...>;
  using index_list =
      tmpl::list<tmpl::at_c<UncontractedIndexList<UncontractedIndices...>,
                            index_maps.first[ContractedInts]>...>;

  using tensorindex_list = tmpl::list<
      tmpl::at_c<UncontractedTensorIndexList<UncontractedTensorIndices...>,
                 index_maps.first[ContractedInts]>...>;
  using type = TensorExpression<UncontractedTensorExpression, DataType,
                                symmetry, index_list, tensorindex_list>;
};
}  // namespace detail

/// \ingroup TensorExpressionsGroup
/// \brief Marks a class as being a `TensorExpressions::TensorContract`
///
/// \details
/// The empty base class provides a simple means for checking if a type is a
/// `TensorExpressions::TensorContract`.
struct MarkAsTensorContract {};

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename T, typename X, typename Symm, typename IndexList,
          typename ArgsList, size_t NumContractedIndices>
struct TensorContract
    : public TensorExpression<
          TensorContract<T, X, Symm, IndexList, ArgsList, NumContractedIndices>,
          X,
          typename detail::ContractedType<
              T, X, Symm, IndexList, ArgsList, NumContractedIndices,
              (tmpl::size<Symm>::value - NumContractedIndices) /
                  2>::type::symmetry,
          typename detail::ContractedType<
              T, X, Symm, IndexList, ArgsList, NumContractedIndices,
              (tmpl::size<Symm>::value - NumContractedIndices) /
                  2>::type::index_list,
          typename detail::ContractedType<
              T, X, Symm, IndexList, ArgsList, NumContractedIndices,
              (tmpl::size<Symm>::value - NumContractedIndices) /
                  2>::type::args_list>,
      MarkAsTensorContract {
  using contracted_type = typename detail::ContractedType<
      T, X, Symm, IndexList, ArgsList, NumContractedIndices,
      (tmpl::size<Symm>::value - NumContractedIndices) / 2>;
  using new_type = typename contracted_type::type;

  using type = X;
  using symmetry = typename new_type::symmetry;
  using index_list = typename new_type::index_list;
  using args_list = typename new_type::args_list;
  static constexpr size_t num_tensor_indices = NumContractedIndices;
  static constexpr size_t num_uncontracted_tensor_indices =
      tmpl::size<Symm>::value;
  // TODO : maybe put in static_asserts in ContractedType to check
  // for consistency of these num_X_indices variables
  static constexpr size_t num_indices_to_contract =
      contracted_type::num_indices_to_contract;
  static_assert(num_indices_to_contract > 0,
                "There are no indices to contract that were found.");
  static_assert(num_indices_to_contract % 2 == 0,
                "Cannot contract an odd number of indices.");
  static constexpr size_t num_contracted_index_pairs =
      contracted_type::num_contracted_index_pairs;
  static constexpr inline std::pair<
      std::array<size_t, NumContractedIndices>,
      std::array<std::pair<size_t, size_t>, num_contracted_index_pairs>>
      index_maps = contracted_type::index_maps;
  static constexpr inline std::array<std::pair<size_t, size_t>,
                                     num_contracted_index_pairs>
      contracted_index_shifts = contracted_type::contracted_index_shifts;
  static constexpr inline std::array<size_t, num_uncontracted_tensor_indices>
      uncontracted_index_dims = contracted_type::uncontracted_index_dims;
  static constexpr size_t num_terms_summed = contracted_type::num_terms_summed;
  static constexpr size_t num_ops_left =
      T::num_ops_subtree * num_terms_summed + num_terms_summed - 1;
  static constexpr size_t num_ops_right = 0;
  static constexpr size_t num_ops_subtree = num_ops_left;
  static constexpr size_t num_addsub_ops_subtree =
      T::num_addsub_ops_subtree * num_terms_summed + num_terms_summed - 1;

  explicit TensorContract(
      const TensorExpression<T, X, Symm, IndexList, ArgsList>& t)
      : t_(~t) {}
  ~TensorContract() override = default;

  // TODO : document this and other new stuff
  static constexpr std::array<size_t, num_uncontracted_tensor_indices>
  get_first_index_to_sum(
      const std::array<size_t, num_tensor_indices>& contracted_multi_index) {
    // TODO : make with std::numeric_limits<size_t>::max() with make_with_value
    std::array<size_t, num_uncontracted_tensor_indices>
        uncontracted_multi_index{};

    // set placeholders for debugging
    for (size_t i = 0; i < num_uncontracted_tensor_indices; i++) {
      uncontracted_multi_index[i] = std::numeric_limits<size_t>::max();
    }

    // fill uncontracted indices
    for (size_t i = 0; i < num_tensor_indices; i++) {
      uncontracted_multi_index[index_maps.first[i]] = contracted_multi_index[i];
    }

    // fill contracted indices
    for (size_t i = 0; i < num_contracted_index_pairs; i++) {
      const size_t first_index_position_in_pair = index_maps.second[i].first;
      const size_t second_index_position_in_pair = index_maps.second[i].second;
      // uncontracted_multi_index[first_index_position_in_pair] =
      //     contracted_index_shifts[i].first;
      // uncontracted_multi_index[second_index_position_in_pair] =
      //     contracted_index_shifts[i].second;
      uncontracted_multi_index[first_index_position_in_pair] =
          uncontracted_index_dims[first_index_position_in_pair] - 1;
      uncontracted_multi_index[second_index_position_in_pair] =
          uncontracted_index_dims[second_index_position_in_pair] - 1;
    }

    return uncontracted_multi_index;
  }

  // TODO : use gsl::at instead of brackets everywhere in this file
  static std::array<size_t, num_uncontracted_tensor_indices>
  get_next_multi_index_to_sum(
      const std::array<size_t, num_uncontracted_tensor_indices>&
          uncontracted_multi_index) {
    std::array<size_t, num_uncontracted_tensor_indices>
        next_uncontracted_multi_index = uncontracted_multi_index;

    size_t i = 0;
    while (i < num_contracted_index_pairs) {
      const size_t current_index_first_position = index_maps.second[i].first;
      const size_t current_index_second_position = index_maps.second[i].second;
      const size_t current_index_first_shift = contracted_index_shifts[i].first;
      // const size_t current_index_second_shift =
      //     contracted_index_shifts[i].second;
      // const size_t current_index_dim_to_contract =
      //     uncontracted_index_dims[current_index_first_position] -
      //     current_index_first_shift;

      // TODO: this will need to account for spatial spacetime shifts
      // next_uncontracted_multi_index[current_index_first_position] =
      // (uncontracted_multi_index[current_index_first_position] + 1) %
      // contracted_index_pair_dims[i];
      // next_uncontracted_multi_index[current_index_second_position] =
      // (uncontracted_multi_index[current_index_second_position] + 1) %
      // contracted_index_pair_dims[i];

      // note: this way works:
      // next_uncontracted_multi_index[current_index_first_position] =
      //     (uncontracted_multi_index[current_index_first_position] + 1) %
      //     uncontracted_index_dims[current_index_first_position];
      // next_uncontracted_multi_index[current_index_second_position] =
      //     (uncontracted_multi_index[current_index_second_position] + 1) %
      //     uncontracted_index_dims[current_index_second_position];
      // if (next_uncontracted_multi_index[current_index_first_position] != 0) {
      //   break;
      // }
      // next_uncontracted_multi_index[current_index_first_position] =
      //     current_index_first_shift;
      // next_uncontracted_multi_index[current_index_second_position] =
      //     current_index_second_shift;

      // TODO : make the rest of this loop more elegant?
      next_uncontracted_multi_index[current_index_first_position]--;
      next_uncontracted_multi_index[current_index_second_position]--;

      // if the next index value is 0 when we have spatial spacetime indices
      // or if the next index value is the max size_t, then we've wrapped around
      // and we need to go again
      if (not(next_uncontracted_multi_index[current_index_first_position] <
                  current_index_first_shift or
              next_uncontracted_multi_index[current_index_first_position] >
                  uncontracted_index_dims[current_index_first_position])) {
        break;
      }

      next_uncontracted_multi_index[current_index_first_position] =
          uncontracted_index_dims[current_index_first_position] - 1;
      next_uncontracted_multi_index[current_index_second_position] =
          uncontracted_index_dims[current_index_second_position] - 1;

      i++;
    }

    return next_uncontracted_multi_index;
  }

  template <size_t Iteration>
  static SPECTRE_ALWAYS_INLINE decltype(auto) compute_contraction(
      const T& t, const std::array<size_t, num_uncontracted_tensor_indices>&
                      current_multi_index) {
    if constexpr (Iteration < num_terms_summed - 1) {
      std::array<size_t, num_uncontracted_tensor_indices> next_multi_index =
          get_next_multi_index_to_sum(current_multi_index);

      // std::cout << "current_multi_index : [ ";
      // for (const size_t elem : current_multi_index) {
      //   std::cout << elem << " ";
      // }
      // std::cout << "]" << std::endl;

      // std::cout << "next_multi_index : [ ";
      // for (const size_t elem : next_multi_index) {
      //   std::cout << elem << " ";
      // }
      // std::cout << "]" << std::endl;

      // We have more than one component left to sum
      return compute_contraction<Iteration + 1>(t, next_multi_index) +
             t.get(current_multi_index);
    } else {
      // std::cout << "BASE current_multi_index: [ ";
      // for (const size_t elem : current_multi_index) {
      //   std::cout << elem << " ";
      // }
      // std::cout << "]" << std::endl;

      // We only have one final component to sum
      return t.get(current_multi_index);
    }
  }

  decltype(auto) get(const std::array<size_t, num_tensor_indices>&
                         contracted_multi_index) const {
    std::array<size_t, num_uncontracted_tensor_indices>
        first_operand_multi_index_to_sum =
            get_first_index_to_sum(contracted_multi_index);
    // std::cout << "first_operand_multi_index_to_sum : \n[";
    // for (auto i : contracted_multi_index) {
    //   std::cout << contracted_multi_index[i] << " ";
    // }
    // std::cout << "]" << std::endl;

    return compute_contraction<0>(t_, first_operand_multi_index_to_sum);
  }

 private:
  T t_;
};

template <typename T, typename X, typename Symm, typename IndexList,
          typename... TensorIndices>
SPECTRE_ALWAYS_INLINE static constexpr auto contract(
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<TensorIndices...>>&
        t) {
  // TODO : update naming everywhere to make "uncontracted_..." make sense
  constexpr size_t num_uncontracted_indices = sizeof...(TensorIndices);
  constexpr size_t num_contracted_index_pairs =
      detail::get_num_contracted_index_pairs<num_uncontracted_indices>(
          {{TensorIndices::value...}});

  if constexpr (num_contracted_index_pairs == 0) {
    // There aren't any indices to contract, so we just return the input
    return ~t;
  } else {
    // We have at least one pair of indices to contract
    return TensorContract<T, X, Symm, IndexList, tmpl::list<TensorIndices...>,
                          num_uncontracted_indices -
                              (num_contracted_index_pairs * 2)>{t};
  }
}
}  // namespace TensorExpressions
