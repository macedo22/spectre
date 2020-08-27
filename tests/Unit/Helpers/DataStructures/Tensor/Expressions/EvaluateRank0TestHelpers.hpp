// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 0 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \tparam Datatype the type of data being stored in the Tensors
/// \param data the data being stored in the Tensors
template <typename Datatype>
void test_evaluate_rank_0(const Datatype& data) {
  Tensor<Datatype> rhs_tensor{};
  rhs_tensor.get() = data;

  auto lhs_tensor = TensorExpressions::evaluate(rhs_tensor());

  CHECK(lhs_tensor.get() == data);
}
