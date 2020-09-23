// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Contract",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Aull{};
  std::iota(Aull.begin(), Aull.end(), 0.0);
  std::cout << "Original tensor before contraction : " << Aull << std::endl
            << std::endl;

  auto Iij = Aull(ti_I, ti_i, ti_j);

  std::cout << "new number of TensorIndexs : "
            << tmpl::size<decltype(Iij)::new_type::args_list>::value
            << std::endl;
  std::cout << "new num_tensor_indices: "
            << decltype(Iij)::new_type::num_tensor_indices << std::endl;

  auto Iij_to_j = TensorExpressions::evaluate<ti_j_t>(Iij);

  std::cout << "Newly contracted tensor : " << Iij_to_j << std::endl;

  for (size_t j = 0; j < 3; j++) {
    double expected_sum = 0.0;
    for (size_t i = 0; i < 3; i++) {
      expected_sum += Aull.get(i, i, j);
    }
    CHECK(Iij_to_j.get(j) == expected_sum);
  }

  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Alull{};
  std::iota(Alull.begin(), Alull.end(), 0.0);

  auto kJij = Alull(ti_k, ti_J, ti_i, ti_j);

  std::cout << "new number of TensorIndexs : "
            << tmpl::size<decltype(kJij)::new_type::args_list>::value
            << std::endl;
  std::cout << "new num_tensor_indices: "
            << decltype(kJij)::new_type::num_tensor_indices << std::endl;

  auto kJij_to_ki = TensorExpressions::evaluate<ti_k_t, ti_i_t>(kJij);

  for (size_t k = 0; k < 3; k++) {
    for (size_t i = 0; i < 3; i++) {
      double expected_sum = 0.0;
      for (size_t j = 0; j < 3; j++) {
        expected_sum += Alull.get(k, j, i, j);
      }
      CHECK(kJij_to_ki.get(k, i) == expected_sum);
    }
  }
}
