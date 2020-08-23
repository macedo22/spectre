// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

template <typename Datatype>
void test_storage_get_rank_0(const Datatype& data) {
  Tensor<Datatype> rhs_tensor{};
  rhs_tensor.get() = data;

  auto evaluated_rhs_tensor = TensorExpressions::evaluate(rhs_tensor());

  CHECK(evaluated_rhs_tensor.get() == rhs_tensor.get());
}
