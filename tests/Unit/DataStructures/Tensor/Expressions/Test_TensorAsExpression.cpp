// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorAsExpressionRank2TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.TensorAsExpression",
                  "[DataStructures][Unit]") {
  // Rank 2
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_2(ti_a, ti_b);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_2(ti_b, ti_a);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_2(ti_a, ti_c);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_2(ti_c, ti_a);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_2(ti_a, ti_i);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_2(ti_i, ti_a);
}
