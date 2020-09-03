// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TestingFrameworkGroup
/// \brief Test that the tensor multi-index of a rank 0 RHS Tensor is an empty
/// array
///
/// \tparam Datatype the type of data being stored in the Tensors
template <typename Datatype>
void test_compute_rhs_tensor_index_rank_0() {
  const Tensor<Datatype> rhs_tensor{};
  const auto R = rhs_tensor();

  const std::array<size_t, 0> index_order = {};

  const std::array<size_t, 0> tensor_multi_index = {};
  CHECK(R.template compute_rhs_tensor_index<0>(index_order, index_order,
                                               tensor_multi_index) ==
        tensor_multi_index);
}
