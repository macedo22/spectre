// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRankN.hpp"
#include "DataStructures/DataVector.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank0.hpp"

namespace TestHelpers::tenex {
void test_evaluate() {
  test_evaluate_rank_0<double>();
  test_evaluate_rank_0<DataVector>();
}
}  // namespace TestHelpers::tenex
