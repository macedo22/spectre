// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace {
constexpr size_t num_points = 8;
const DataVector used_for_size =
    DataVector(num_points, std::numeric_limits<double>::signaling_NaN());
constexpr size_t Dim = 1;
using FrameType = Frame::Inertial;
using R_type = Tensor<DataVector, Symmetry<2, 1>,
                      index_list<SpacetimeIndex<Dim, UpLo::Up, FrameType>,
                                 SpacetimeIndex<Dim, UpLo::Up, FrameType>>>;
using S_type = Tensor<DataVector, Symmetry<2, 1>,
                      index_list<SpacetimeIndex<Dim, UpLo::Lo, FrameType>,
                                 SpacetimeIndex<Dim, UpLo::Lo, FrameType>>>;

template <typename DataVector>
void test(gsl::not_null<Scalar<DataVector>*> L, const R_type& R,
          const S_type& S) {
  const size_t initial_num_allocations =
      VectorImpl<double, DataVector>::get_num_allocations();
  // 10
  std::cout << "VectorImpl<double, DataVector>::num_allocations_ (initial) : "
            << initial_num_allocations << std::endl;

  TensorExpressions::evaluate(L, R(ti_A, ti_B) * S(ti_a, ti_b));

  const size_t final_num_allocations =
      VectorImpl<double, DataVector>::get_num_allocations();
  // 10
  std::cout << "VectorImpl<double, DataVector>::num_allocations_ (final) : "
            << final_num_allocations << std::endl;

  const size_t num_allocations_this_function =
      final_num_allocations - initial_num_allocations;
  // 0
  std::cout << "VectorImpl<double, DataVector>::num_allocations_ from "
               "evaluating LHS tensor : "
            << num_allocations_this_function << std::endl;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerProduct2x2TE",
                  "[Unit][DataStructures]") {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto R = make_with_random_values<R_type>(make_not_null(&generator),
                                                 distribution, used_for_size);
  const auto S = make_with_random_values<S_type>(make_not_null(&generator),
                                                 distribution, used_for_size);

  Scalar<DataVector> L(used_for_size);
  test(make_not_null(&L), R, S);
}
