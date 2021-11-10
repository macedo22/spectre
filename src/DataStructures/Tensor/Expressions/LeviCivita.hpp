// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Levi-Civita symbols used in `TensorExpression`s

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct LeviCivitaSymbol {};
}  // namespace Frame

namespace TensorExpressions {
namespace detail {
template <size_t Dim, typename IndexSequence = std::make_index_sequence<Dim>>
struct LeviCivitaSymbolSymm;

template <size_t Dim, std::int32_t... Ints>
struct LeviCivitaSymbolSymm<Dim, std::integer_sequence<Ints...>> {
  using symm = tmpl::integral_list<std::int32_t, (Dim - Ints)...>;
};
}  // namespace detail

template <size_t Dim, Requires<(Dim > 1)> = nullptr>
constexpr bool is_even_permutation(const std::array<size_t, Dim>& multi_index) {
  const size_t first_index_value = gsl::at(multi_index, 0);
  for (size_t i = 1; i < Dim; i++) {
    const size_t other_index_value = gsl::at(multi_index, i);
    if (other_index_value != ((first_index_value + i) % Dim)) {
      return false;
    }
  }
  return true;
}

template <size_t Dim>
constexpr bool contains_repeated_concrete_index_value(
    const std::array<size_t, Dim>& multi_index) {
  if constexpr (Dim < 2) {
    return false;
  } else {
    for (size_t i = 0; i < Dim - 1; i++) {
      const size_t current_index_value = gsl::at(multi_index, i);
      for (size_t j = i + 1; j < Dim; j++) {
        const size_t other_index_value = gsl::at(multi_index, j);
        if (current_index_value == other_index_value) {
          return true;
        }
      }
    }
    return false;
  }
}

template <typename... TensorIndices>
struct LeviCivitaSymbolExpressionOperand;

template <size_t Dim>
struct LeviCivitaSymbol {
  static constexpr std::make_integer_sequence<std::int32_t, Dim> {}
  seq{};
  using symmetry = LeviCivitaSymbolSymm<Dim, seq>::symmetry;
  static constexpr size_t dim = Dim;

  template <typename... TensorIndices>
  SPECTRE_ALWAYS_INLINE constexpr auto operator()(
      TensorIndices... /*meta*/) const {
    static_assert(
        (... and tt::is_tensor_index<TensorIndices>::value),
        "A Levi-Civita symbol expression must be created using TensorIndex "
        "objects to represent generic indices, e.g. ti_i, ti_j, "
        "etc.");
    static_assert(
        (... and (not(tt::is_time_index<TensorIndices>::value))),
        "A Levi-Civita symbol expression cannot be created using time "
        "indices.");
    static_assert(
        detail::tensorindex_list_is_valid<TensorIndices...>::value,
        "Cannot create a Levi-Civita symbol expression with a repeated generic "
        "index.");
    static_assert(
        not detail::contains_indices_to_contract<Dim>(
            {{TensorIndices::value...}}),
        "Cannot create a Levi-Civita symbol expression with contractible "
        "indices.");
    static_assert(
        detail::tensorindices_same_indextype<TensorIndices...>::value),
        "The TensorIndexs used to create a Levi-Civita symbol expression must "
        "either be all spatial or all spacetime.");
    return LeviCivitaSymbolExpressionOperand<TensorIndices...>{*this};
  }
};

struct MarkAsLeviCivitaSymbolExpressionOperand {};

template <typename L, typename... TensorIndices>
struct LeviCivitaSymbolExpressionOperand
    : public MarkAsLeviCivitaSymbolExpressionOperand {
  using type = double;
  using symmetry = typename L::Symm;
  using index_list =
      index_list<Tensor_detail::TensorIndexType<Dim, TensorIndices::valence,
                                                Frame::LeviCivitaSymbol,
                                                TensorIndices::indextype>...>;
  using args_list = tmpl::list<TensorIndices...>;
  static constexpr size_t num_tensor_indices = L::dim;

  LeviCivitaSymbolExpressionOperand(const L& l) : l_(&l) {}

 private:
  const L* l_ = nullptr;
};

template <typename L, typename... TensorIndices>
struct LevitCivitaSymbolInnerProduct {};

template <typename L, typename T>
struct LeviCivitaSymbolOuterProduct
    : public TensorExpression<
          LeviCivitaSymbolOuterProduct<L, T>,
          typename detail::OuterProductType<L, T>::type,
          typename detail::OuterProductType<L, T>::symmetry,
          typename detail::OuterProductType<L, T>::index_list,
          typename detail::OuterProductType<L, T>::tensorindex_list> {
  using type = typename detail::OuterProductType<T1, T2>::type;
  using symmetry = typename detail::OuterProductType<T1, T2>::symmetry;
  using index_list = typename detail::OuterProductType<T1, T2>::index_list;
  using args_list = typename detail::OuterProductType<T1, T2>::tensorindex_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  static constexpr auto l_num_tensor_indices =
      tmpl::size<typename L::index_list>::value;
  static constexpr auto t_num_tensor_indices =
      num_tensor_indices - l_num_tensor_indices;

  LeviCivitaSymbolOuterProduct(L l, T t) : l_(std::move(l)), t_(std::move(t)) {}
  ~LeviCivitaSymbolOuterProduct() override = default;

  SPECTRE_ALWAYS_INLINE type
  get(const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    std::array<size_t, l_num_tensor_indices> levi_civita_multi_index{};
    for (size_t i = 0; i < l_num_tensor_indices; i++) {
      gsl::at(levi_civita_multi_index, i) = gsl::at(result_multi_index, i);
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
      //    different LeviCivitaSymbolOuterProduct.get() s. Or have this have a
      //    parent expression that is hit first, so then this choice can be made
      //    there instead of here. However, this will probably still have issues
      //    with decltype(auto) in upstream get()s.
      //
      //    All options ^ have a decltype(auto) problem here or upstream
      return 0.0 * t_.get(tensorexpression_mutli_index);
    } else {
      std::array<size_t, t_num_tensor_indices> tensorexpression_multi_index{};
      for (size_t i = 0; i < t_num_tensor_indices; i++) {
        gsl::at(tensorexpression_multi_index, i) =
            gsl::at(result_multi_index, l_num_tensor_indices + i);
      }

      if (is_even_permutation(levi_civita_multi_index)) {
        return t_.get(tensorexpression_mutli_index);
      } else {
        return -t_.get(tensorexpression_mutli_index);
      }
    }
  }

 private:
  L l_;
  T t_;
};
}  // namespace TensorExpressions

static constexpr TensorExpressions::LeviCivitaSymbol<2> te_levi_civita_2d{};
static constexpr TensorExpressions::LeviCivitaSymbol<3> te_levi_civita_3d{};
static constexpr TensorExpressions::LeviCivitaSymbol<4> te_levi_civita_4d{};
