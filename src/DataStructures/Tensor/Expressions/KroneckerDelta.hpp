// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines TODO

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
using KroneckerDeltaFrame = Frame::NoFrame;

// forward declaration
template <typename K, typename TensorIndex1, typename TensorIndex2>
struct KroneckerDeltaAsExpression;

struct MarkAsKroneckerDelta {};

template <size_t Dim>
struct KroneckerDelta : public MarkAsKroneckerDelta {
  using symmetry = Symmetry<2, 1>;
  static constexpr size_t dim = Dim;

  template <typename TensorIndex1, typename TensorIndex2>
  SPECTRE_ALWAYS_INLINE constexpr auto operator()(TensorIndex1 /*meta*/,
                                                  TensorIndex2 /*meta*/) const {
    static_assert(
        tt::is_tensor_index<TensorIndex1>::value and
            tt::is_tensor_index<TensorIndex2>::value,
        "A Kronecker delta expression must be created using TensorIndex "
        "objects to represent generic indices, e.g. ti::I, ti::j.");
    static_assert(
        not tt::is_time_index<TensorIndex1>::value and
            not tt::is_time_index<TensorIndex2>::value,
        "A Kronecker delta expression cannot be created using time indices.");
    static_assert(
        TensorIndex1::valence != TensorIndex2::valence,
        "Kronecker delta expressions needs to be be created using one upper "
        "index and one lower index.");
    static_assert(
        TensorIndex1::is_spacetime == TensorIndex2::is_spacetime,
        "The TensorIndexs used to create a Kronecker delta expression must "
        "either be both spatial or both spacetime.");

    if constexpr (get_tensorindex_value_with_opposite_valence(
                      TensorIndex1::value) != TensorIndex2::value) {
      return KroneckerDeltaAsExpression<KroneckerDelta<Dim>, TensorIndex1,
                                        TensorIndex2>{*this};
    } else {
      // trace of Kronecker delta = dim
      return NumberAsExpression(1.0 * Dim);
    }
  }
};
}  // namespace tenex

static constexpr tenex::KroneckerDelta<1> kdelta1{};
static constexpr tenex::KroneckerDelta<2> kdelta2{};
static constexpr tenex::KroneckerDelta<3> kdelta3{};
