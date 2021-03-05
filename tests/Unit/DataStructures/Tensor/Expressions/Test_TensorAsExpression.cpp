// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorAsExpressionRank2TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorAsExpressionRank3TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.TensorAsExpression",
                  "[DataStructures][Unit]") {
  // Rank 2
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_2(ti_i, ti_j);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_2(ti_j, ti_i);

  // Rank 3
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_3(ti_a, ti_b,
                                                                   ti_c);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_3(ti_a, ti_c,
                                                                   ti_b);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_3(ti_b, ti_a,
                                                                   ti_c);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_3(ti_b, ti_c,
                                                                   ti_a);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_3(ti_c, ti_a,
                                                                   ti_b);
  TestHelpers::TensorExpressions::test_tensor_as_expression_rank_3(ti_c, ti_b,
                                                                   ti_a);
}
