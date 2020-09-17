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
/// \tparam DataType the type of data being stored in the Tensors
/// \param data the data being stored in the Tensors
template <typename DataType>
void test_evaluate_rank_0(const DataType& data) {
  const Tensor<DataType> R{{{data}}};

  const auto L = TensorExpressions::evaluate(R());

  CHECK(L.get() == data);
}
