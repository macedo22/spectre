// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Kronecker delta objects that can be used in a `TensorExpression`

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"

namespace tenex {
template <typename K, typename TensorIndex1, typename TensorIndex2>
struct KroneckerDeltaAsExpression;

/// \ingroup TensorExpressionsGroup
/// \brief Defines Kronecker delta objects that can be used in a
/// `TensorExpression`
///
/// \details
/// This is not a derived `TensorExpression` type, but when used with generic
/// indices, you get a derived `TensorExpression` representing the Kronecker
/// delta with those generic indices. Put another way, `KroneckerDelta` is to
/// `KroneckerDeltaAsExpression` as `Tensor` is to `TensorAsExpression`.
///
/// \tparam Dim the dimension of the Kronecker delta
template <size_t Dim, typename FrameType>
struct KroneckerDelta {
  /// The ::Symmetry of the Kronecker delta
  using symmetry = Symmetry<2, 1>;
  /// The ::Frame of the Kronecker delta
  using Frame = FrameType;
  /// The dimension of the Kronecker delta
  static constexpr size_t dim = Dim;

  /// \brief Retrieve a `TensorExpression` object with the specified generic
  /// indices
  ///
  /// \tparam TensorIndex1 the first \ref TensorIndex "generic index"
  /// \tparam TensorIndex2 the second \ref TensorIndex "generic index"
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
        "Kronecker delta expressions need to be be created using one upper "
        "index and one lower index.");
    static_assert(
        TensorIndex1::is_spacetime == TensorIndex2::is_spacetime,
        "The TensorIndexs used to create a Kronecker delta expression must "
        "either be both spatial or both spacetime generic indices.");

    if constexpr (get_tensorindex_value_with_opposite_valence(
                      TensorIndex1::value) != TensorIndex2::value) {
      // don't contract indices
      return KroneckerDeltaAsExpression<KroneckerDelta<Dim, Frame>,
                                        TensorIndex1, TensorIndex2>{*this};
    } else {
      // contract indices, where trace of Kronecker delta = dim
      return NumberAsExpression(1.0 * Dim);
    }
  }
};
}  // namespace tenex

/// \ingroup TensorExpressionsGroup
/// \brief The Kronecker delta objects to use in a `TensorExpression`
///
/// \details
/// To use in a `TensorExpression`, you must supply one upper and one lower
/// generic index (`TensorIndex`), e.g.:
///
/// \code
/// // 3D Kronecker delta in the inertial frame
/// kronecker_delta<3>(ti::I, ti::j)
/// // 2D Kronecker delta in the grid frame
/// kronecker_delta<2, Frame::Grid>(ti::I, ti::j)
/// \endcode
///
/// The upper and lower index can be in any order, but both need to be generic
/// spatial indices or generic spacetime indices. Examples:
///
/// \code
/// kronecker_delta<3>(ti::I, ti::j)  // OK
/// kronecker_delta<3>(ti::i, ti::J)  // OK
/// kronecker_delta<3>(ti::A, ti::b)  // OK
/// kronecker_delta<3>(ti::J, ti::j)  // OK, not advised since its just the dim
///
/// kronecker_delta<3>(ti::I, ti::J)  // ERROR: both upper indices
/// kronecker_delta<3>(ti::I, ti::a)  // ERROR: spatial and spacetime index
/// kronecker_delta<3>(ti::I, ti::t)  // ERROR: concrete index (time index)
/// \endcode
template <size_t Dim, typename FrameType = Frame::Inertial>
constexpr tenex::KroneckerDelta<Dim, FrameType> kronecker_delta{};
