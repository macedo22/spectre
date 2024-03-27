// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace {
DataVector triple_product(const DataVector& a, const DataVector& b,
                          const DataVector& c, const DataVector& d,
                          const DataVector& e, const DataVector& f) {
  return (a + b) * (c + d) * (e + f);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataVectorExpressions",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  constexpr size_t num_points = 1000;
  const DataVector used_for_size =
      DataVector(num_points, std::numeric_limits<double>::signaling_NaN());

  DataVector a = make_with_random_values<DataVector>(
      make_not_null(&generator), distribution, used_for_size);
  DataVector b = make_with_random_values<DataVector>(
      make_not_null(&generator), distribution, used_for_size);
  DataVector c = make_with_random_values<DataVector>(
      make_not_null(&generator), distribution, used_for_size);
  DataVector d = make_with_random_values<DataVector>(
      make_not_null(&generator), distribution, used_for_size);
  DataVector e = make_with_random_values<DataVector>(
      make_not_null(&generator), distribution, used_for_size);
  DataVector f = make_with_random_values<DataVector>(
      make_not_null(&generator), distribution, used_for_size);

  DataVector result = triple_product(a, b, c, d, e, f);

  //   std::cout << a << std::endl;
  //   std::cout << b << std::endl;

  CHECK(result.size() == num_points);
}
