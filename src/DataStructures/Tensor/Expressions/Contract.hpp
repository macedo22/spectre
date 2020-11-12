// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Expression Templates for contracting tensor indices on a single
/// Tensor

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Requires.hpp"

/*!
 * \ingroup TensorExpressionsGroup
 * Holds all possible TensorExpressions currently implemented
 */
namespace TensorExpressions {

namespace detail {

template <typename I1, typename I2>
using indices_contractible = std::integral_constant<
    bool, I1::dim == I2::dim and I1::ul != I2::ul and
              std::is_same_v<typename I1::Frame, typename I2::Frame> and
              I1::index_type == I2::index_type>;

template <typename T, typename X, typename SymmList, typename IndexList,
          typename Args>
struct ContractedTypeImpl;

template <typename T, typename X, template <typename...> class SymmList,
          typename IndexList, typename Args, typename... Symm>
struct ContractedTypeImpl<T, X, SymmList<Symm...>, IndexList, Args> {
  using type =
      TensorExpression<T, X, Symmetry<Symm::value...>, IndexList, Args>;
};

template <typename ReplacedArg1, typename ReplacedArg2, typename T, typename X,
          typename Symm, typename IndexList, typename Args>
struct ContractedType {
  using contracted_symmetry = tmpl::erase<
      tmpl::erase<Symm, tmpl::index_of<Args, ReplacedArg2>>,
      tmpl::index_of<tmpl::erase<Args, tmpl::index_of<Args, ReplacedArg2>>,
                     ReplacedArg1>>;
  using contracted_index_list = tmpl::erase<
      tmpl::erase<IndexList, tmpl::index_of<Args, ReplacedArg2>>,
      tmpl::index_of<tmpl::erase<Args, tmpl::index_of<Args, ReplacedArg2>>,
                     ReplacedArg1>>;
  using contracted_tensorindex_list = tmpl::erase<
      tmpl::erase<Args, tmpl::index_of<Args, ReplacedArg2>>,
      tmpl::index_of<tmpl::erase<Args, tmpl::index_of<Args, ReplacedArg2>>,
                     ReplacedArg1>>;
  using type = typename ContractedTypeImpl<T, X, contracted_symmetry,
                                           contracted_index_list,
                                           contracted_tensorindex_list>::type;
};

template <size_t I, size_t Index1, size_t Index2, typename... LhsIndices,
          typename T, typename S>
static SPECTRE_ALWAYS_INLINE decltype(auto) compute_contraction(S tensor_index,
                                                                const T& t1) {
  if constexpr (I == 0) {
    tensor_index[Index1] = 0;
    tensor_index[Index2] = 0;
    return t1.template get<LhsIndices...>(tensor_index);
  } else {
    tensor_index[Index1] = I;
    tensor_index[Index2] = I;
    return t1.template get<LhsIndices...>(tensor_index) +
           compute_contraction<I - 1, Index1, Index2, LhsIndices...>(
               tensor_index, t1);
  }
}
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename ReplacedArg1, typename ReplacedArg2, typename T, typename X,
          typename Symm, typename IndexList, typename ArgsList>
struct TensorContract
    : public TensorExpression<TensorContract<ReplacedArg1, ReplacedArg2, T, X,
                                             Symm, IndexList, ArgsList>,
                              X,
                              typename detail::ContractedType<
                                  ReplacedArg1, ReplacedArg2, T, X, Symm,
                                  IndexList, ArgsList>::type::symmetry,
                              typename detail::ContractedType<
                                  ReplacedArg1, ReplacedArg2, T, X, Symm,
                                  IndexList, ArgsList>::type::index_list,
                              typename detail::ContractedType<
                                  ReplacedArg1, ReplacedArg2, T, X, Symm,
                                  IndexList, ArgsList>::type::args_list> {
  static constexpr size_t replaced_arg1_position =
      tmpl::index_of<ArgsList, ReplacedArg1>::value;
  static constexpr size_t replaced_arg2_position =
      tmpl::index_of<ArgsList, ReplacedArg2>::value;

  // "first" and "second" here refer to the position of the indices to contract
  // in the list of generic indices, with "first" denoting leftmost
  //
  // e.g. `R(ti_A, ti_b, ti_a)` :
  // - `first_replaced_tensorindex` refers to the contracted TensorIndex type
  //   for `ti_A`
  // - `second_replaced_tensorindex` refers to the contracted TensorIndex type
  //   for `ti_a`
  static constexpr size_t first_position_to_contract =
      (replaced_arg1_position < replaced_arg2_position)
          ? replaced_arg1_position
          : replaced_arg2_position;
  static constexpr size_t second_position_to_contract =
      (replaced_arg1_position < replaced_arg2_position)
          ? replaced_arg2_position
          : replaced_arg1_position;

  // first and second \ref SpacetimeIndex "TensorIndexType"s to contract
  using CI1 = tmpl::at_c<IndexList, first_position_to_contract>;
  using CI2 = tmpl::at_c<IndexList, second_position_to_contract>;
  static_assert(tmpl::size<Symm>::value > 1 and
                    tmpl::size<IndexList>::value > 1,
                "Cannot contract indices on a Tensor with rank less than 2");
  static_assert(detail::indices_contractible<CI1, CI2>::value,
                "Cannot contract the requested indices.");

  using new_type =
      typename detail::ContractedType<ReplacedArg1, ReplacedArg2, T, X, Symm,
                                      IndexList, ArgsList>::type;

  using type = X;
  using symmetry = typename new_type::symmetry;
  using index_list = typename new_type::index_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  using args_list = typename new_type::args_list;

  explicit TensorContract(
      const TensorExpression<T, X, Symm, IndexList, ArgsList>& t)
      : t_(~t) {}

  template <size_t I, size_t Rank>
  SPECTRE_ALWAYS_INLINE void fill_contracting_tensor_index(
      std::array<size_t, Rank>& tensor_index_in,
      const std::array<size_t, num_tensor_indices>& tensor_index) const {
    if constexpr (I < first_position_to_contract) {
      tensor_index_in[I] = tensor_index[I];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I == first_position_to_contract) {
      // 10000 is for the slot that will be set later. Easy to debug.
      tensor_index_in[I] = 10000;
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I > first_position_to_contract and
                         I <= second_position_to_contract and I < Rank - 1) {
      // tensor_index is Rank - 2 since it shouldn't be called for Rank 2 case
      // 20000 is for the slot that will be set later. Easy to debug.
      tensor_index_in[I] =
          I == second_position_to_contract ? 20000 : tensor_index[I - 1];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I > second_position_to_contract and I < Rank - 1) {
      // Left as Rank - 2 since it should never be called for the Rank 2 case
      tensor_index_in[I] = tensor_index[I - 2];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I == second_position_to_contract) {
      tensor_index_in[I] = 20000;
    } else {
      tensor_index_in[I] = tensor_index[I - 2];
    }
  }

  template <typename... LhsIndices, typename U>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<U, num_tensor_indices>& new_tensor_index) const {
    // new_tensor_index is the one with _fewer_ components, ie post-contraction
    std::array<size_t, tmpl::size<Symm>::value> tensor_index{};
    // Manually unrolled for loops to compute the tensor_index from the
    // new_tensor_index
    fill_contracting_tensor_index<0>(tensor_index, new_tensor_index);
    return detail::compute_contraction<CI1::dim - 1, first_position_to_contract,
                                       second_position_to_contract,
                                       LhsIndices...>(tensor_index, t_);
  }

  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const size_t lhs_storage_index) const {
    const std::array<size_t, num_tensor_indices>& new_tensor_index =
        LhsStructure::template get_canonical_tensor_index<num_tensor_indices>(
            lhs_storage_index);
    return get<LhsIndices...>(new_tensor_index);
  }

 private:
  const std::conditional_t<std::is_base_of<Expression, T>::value, T,
                           TensorExpression<T, X, Symm, IndexList, ArgsList>>
      t_;
};

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename RepeatedTensorIndexValueList, typename T, typename X,
          typename Symm, typename IndexList, typename ReplacedTensorIndexList>
SPECTRE_ALWAYS_INLINE static constexpr auto contract(
    const TensorExpression<T, X, Symm, IndexList, ReplacedTensorIndexList>&
        t) noexcept {
  if constexpr (tmpl::size<RepeatedTensorIndexValueList>::value == 0) {
    // There aren't any repeated indices, so we just return the input
    return ~t;
  } else {
    // We have repeated indices so we must contract
    constexpr size_t index_value_to_contract =
        tmpl::front<RepeatedTensorIndexValueList>::value;
    using lower_tensorindex =
        ti_contracted_t<TensorIndex<index_value_to_contract, UpLo::Lo>>;
    using upper_tensorindex =
        ti_contracted_t<TensorIndex<index_value_to_contract, UpLo::Up>>;

    return contract<tmpl::pop_front<RepeatedTensorIndexValueList>>(
        TensorContract<lower_tensorindex, upper_tensorindex, T, X, Symm,
                       IndexList, ReplacedTensorIndexList>{t});
  }
}
}  // namespace TensorExpressions
