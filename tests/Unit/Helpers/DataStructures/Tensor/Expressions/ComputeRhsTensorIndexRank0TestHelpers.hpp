// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

// Check mapping for all positions from 0 to spatial_dim
template <typename Datatype>
void test_compute_rhs_tensor_index_rank_0() {
  Tensor<Datatype> rhs_tensor{};

  auto rhs_tensor_expr = rhs_tensor();

  std::array<size_t, 0> index_order = {};

  const std::array<size_t, 0> arr = {};
  CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<0>(
        index_order, index_order, arr) == arr);
}
