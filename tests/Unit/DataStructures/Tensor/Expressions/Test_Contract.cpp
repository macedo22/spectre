// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Contract",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Aul{};
  std::iota(Aul.begin(), Aul.end(), 0.0);

  auto Aul_exp = Aul(ti_A, ti_a);

  /*auto tensor = TensorExpressions::evaluate(Aul_exp);
  auto result = tensor.get();

  double sum = 0.0;
  for (int i = 0; i < 3; ++i) {
    sum += Aul.get(i, i);
  }

  CHECK(sum == result);*/
}
