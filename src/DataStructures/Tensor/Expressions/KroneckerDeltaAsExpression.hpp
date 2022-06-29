// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines TODO

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
namespace detail {
template <typename TransformedIndices, typename IndexToTransform,
          typename TargetFrame>
struct replace_kronecker_delta_frame_impl {
  using type =
      tmpl::push_back<TransformedIndices,
                      change_index_frame<IndexToTransform, TargetFrame>>;
};

// when the other TE has a different frame type, use that
template <typename K, typename TargetFrame>
struct replace_kronecker_delta_frame_helper {
  using type = tmpl::fold<K::index_list, tmpl::list<>,
                          replace_kronecker_delta_frame_impl<
                              tmpl::_state, tmpl::_element, TargetFrame>>;
};

// whent the other TE also is KroneckerDeltaFrame, don't transform K's
// index_list
template <typename K>
struct replace_kronecker_delta_frame_helper<K, KroneckerDeltaFrame> {
  using type = typename K::index_list;
};

// TODO : note that this assumes the frames of the K indices are the same,
// which they should always be
template <typename K, typename T>
using replace_kronecker_delta_frame = replace_kronecker_delta_frame_helper<
    K, KroneckerDeltaOuterProductFrame<T>::type>::type;
}  // namespace detail

struct MarkAsKroneckerDeltaAsExpression {};

template <typename K, typename TensorIndex1, typename TensorIndex2>
struct KroneckerDeltaAsExpression : public MarkAsKroneckerDeltaAsExpression {
  // === Index properties ===
  /// The type of the data being stored in the result of the expression
  using type = double;
  /// The ::Symmetry of the result of the expression
  using symmetry = Symmetry<2, 1>;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using index_list =
      index_list<Tensor_detail::TensorIndexType<K::dim, TensorIndex1::valence,
                                                KroneckerDeltaFrame,
                                                TensorIndex1::indextype>,
                 Tensor_detail::TensorIndexType<K::dim, TensorIndex2::valence,
                                                KroneckerDeltaFrame,
                                                TensorIndex2::indextype>>;
  /// The list of generic `TensorIndex`s of the result of the expression
  using args_list = tmpl::list<TensorIndex1, TensorIndex2>;
  /// The number of tensor indices in the result of the expression
  static constexpr size_t num_tensor_indices = 2;

  // === Arithmetic tensor operations properties ===
  /// The number of arithmetic tensor operations done in the subtree for the
  /// left operand, which is 0 because this is a leaf expression
  static constexpr size_t num_ops_left_child = 0;
  /// The number of arithmetic tensor operations done in the subtree for the
  /// right operand, which is 0 because this is a leaf expression
  static constexpr size_t num_ops_right_child = 0;
  /// The total number of arithmetic tensor operations done in this expression's
  /// whole subtree, which is 0 because this is a leaf expression
  static constexpr size_t num_ops_subtree = 0;

  // === Properties for splitting up subexpressions along the primary path ===
  // These definitions only have meaning if this expression actually ends up
  // being along the primary path that is taken when evaluating the whole tree.
  // See documentation for `TensorExpression` for more details.
  /// If on the primary path, whether or not the expression is an ending point
  /// of a leg
  static constexpr bool is_primary_end = true;
  /// If on the primary path, this is the remaining number of arithmetic tensor
  /// operations that need to be done in the subtree of the child along the
  /// primary path, given that we will have already computed the whole subtree
  /// at the next lowest leg's starting point. This is just 0 because this
  /// expression is a leaf.
  static constexpr size_t num_ops_to_evaluate_primary_left_child = 0;
  /// If on the primary path, this is the remaining number of arithmetic tensor
  /// operations that need to be done in the right operand's subtree. This is
  /// just 0 because this expression is a leaf.
  static constexpr size_t num_ops_to_evaluate_primary_right_child = 0;
  /// If on the primary path, this is the remaining number of arithmetic tensor
  /// operations that need to be done for this expression's subtree, given that
  /// we will have already computed the subtree at the next lowest leg's
  /// starting point. This is just 0 because this expression is a leaf.
  static constexpr size_t num_ops_to_evaluate_primary_subtree = 0;
  /// If on the primary path, whether or not the expression is a starting point
  /// of a leg
  static constexpr bool is_primary_start = false;
  /// If on the primary path, whether or not the expression's child along the
  /// primary path is a subtree that contains a starting point of a leg along
  /// the primary path. This is always falls because this expression is a leaf.
  static constexpr bool primary_child_subtree_contains_primary_start = false;
  /// If on the primary path, whether or not this subtree contains a starting
  /// point of a leg along the primary path
  static constexpr bool primary_subtree_contains_primary_start =
      is_primary_start;

  KroneckerDeltaAsExpression(const K& k) : k_(&k) {}

 private:
  const K* k_ = nullptr;
};
}  // namespace tenex
