// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines TODO

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Expressions/KroneckerDelta.hpp"
#include "DataStructures/Tensor/Expressions/KroneckerDeltaAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
struct MarkAsKroneckerDeltaOuterProduct {};

template <typename K, typename T>
struct KroneckerDeltaOuterProduct
    : public TensorExpression<
          // TODO : pass in K with index_list frame already transformed by T,
          // then we can also just use detail::OuterProductType as is
          //
          // maybe have this call a helper that first determines target frame
          // if other expression isn't also kronecker delta, then transforms
          // if necessary, then calls detail::OuterProductType
          KroneckerDeltaOuterProduct<K, T>,
          typename detail::OuterProductType<K, T>::type,
          typename detail::OuterProductType<K, T>::symmetry,
          typename detail::OuterProductType<K, T>::index_list,
          typename detail::OuterProductType<K, T>::tensorindex_list>,
      MarkAsKroneckerDeltaOuterProduct {
  // === Index properties ===
  /// The type of the data being stored in the result of the expression
  using type = typename detail::OuterProductType<K, T>::type;
  /// The ::Symmetry of the result of the expression
  using symmetry = typename detail::OuterProductType<K, T>::symmetry;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using index_list = typename detail::OuterProductType<K, T>::index_list;
  /// The list of generic `TensorIndex`s of the result of the
  /// expression
  using args_list = typename detail::OuterProductType<K, T>::tensorindex_list;
  /// The number of tensor indices in the result of the expression
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  /// The number of tensor indices in the Kronecker delta operand
  static constexpr auto num_tensor_indices_k = 2;
  /// The number of tensor indices in the TensorExpression operand
  static constexpr auto num_tensor_indices_t =
      num_tensor_indices - num_tensor_indices_k;

  KroneckerDeltaOuterProduct(K k, T t) : k_(std::move(k)), t_(std::move(t)) {}
  ~KroneckerDeltaOuterProduct() override = default;

  SPECTRE_ALWAYS_INLINE type
  get(const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    std::array<size_t, l_num_tensor_indices> levi_civita_multi_index{};
    for (size_t i = 0; i < l_num_tensor_indices; i++) {
      gsl::at(levi_civita_multi_index, i) = gsl::at(result_multi_index, i);
    }

    std::array<size_t, t_num_tensor_indices> tensorexpression_multi_index{};
    for (size_t i = 0; i < t_num_tensor_indices; i++) {
      gsl::at(tensorexpression_multi_index, i) =
          gsl::at(result_multi_index, l_num_tensor_indices + i);
    }

    if (contains_repeated_concrete_index_value(levi_civita_multi_index)) {
      // TODO: look into std::variant and std::optional and use types like:
      //    double, blaze expression type for t_.get() and blaze expression type
      //    -t_get()? Problem is that std::variant would probably need to be
      //    propagated up.
      //
      //    Could maybe do a single DataVector(0, 0, ...) allocation but then
      //    there will probably still be issues with decltype(auto) with if/else
      //
      //    Another option - have each other class do the index check and call
      //    different KroneckerDeltaOuterProduct.get() s. Or have this have a
      //    parent expression that is hit first, so then this choice can be made
      //    there instead of here. However, this will probably still have issues
      //    with decltype(auto) in upstream get()s.
      //
      //    All options ^ have a decltype(auto) problem here or upstream
      return 0.0 * t_.get(tensorexpression_multi_index);
    } else if (is_even_permutation(levi_civita_multi_index)) {
      return t_.get(tensorexpression_multi_index);
    } else {
      return -t_.get(tensorexpression_multi_index);
    }
  }

 private:
  K k_;
  T t_;
};
}  // namespace tenex

template <size_t LeviCivitaDim, typename T, typename... LeviCivitaTensorIndices>
SPECTRE_ALWAYS_INLINE auto operator*(
    const tenex::LeviCivitaSymbolExpressionOperand<
        tenex::LeviCivitaSymbol<LeviCivitaDim>, LeviCivitaTensorIndices...>& k,
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t) {
  return tenex::contract(tenex::KroneckerDeltaOuterProduct<
                         tenex::LeviCivitaSymbolExpressionOperand<
                             tenex::LeviCivitaSymbol<LeviCivitaDim>,
                             LeviCivitaTensorIndices...>,
                         T>(k, ~t));
}

template <size_t LeviCivitaDim, typename T, typename... LeviCivitaTensorIndices>
SPECTRE_ALWAYS_INLINE auto operator*(
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t,
    const tenex::LeviCivitaSymbolExpressionOperand<
        tenex::LeviCivitaSymbol<LeviCivitaDim>, LeviCivitaTensorIndices...>&
        l) {
  return tenex::contract(tenex::KroneckerDeltaOuterProduct<
                         tenex::LeviCivitaSymbolExpressionOperand<
                             tenex::LeviCivitaSymbol<LeviCivitaDim>,
                             LeviCivitaTensorIndices...>,
                         T>(k, ~t));
}

// TODO: add overloads for doubles

// TODO: add overloads that reorganize tree when multiple outer products
