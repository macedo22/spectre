// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

template <typename TensorIndexList, size_t Index>
size_t get_value_of_tensor_index() {
  return tmpl::at_c<TensorIndexList, Index>::value;
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Contract",
                  "[DataStructures][Unit]") {
  using args_list = tmpl::list<ti_A_t, ti_a_t, ti_b_t, ti_c_t>;
  using repeated_args_list = repeated<args_list>;
  using replaced_args_list = replace_indices<args_list, repeated_args_list>;

  //std::array<size_t, 1> repeated_args_list_arr =
  //    {{repeated_args_list::value...}};
  std::cout << "repeated_args_list[0]: " <<
      get_value_of_tensor_index<repeated_args_list, 0>() << std::endl;
  std::cout << "replaced_args_list[0]: " <<
      get_value_of_tensor_index<replaced_args_list, 0>() << std::endl;
  std::cout << "replaced_args_list[1]: " <<
      get_value_of_tensor_index<replaced_args_list, 1>() << std::endl;
  std::cout << "replaced_args_list[2]: " <<
      get_value_of_tensor_index<replaced_args_list, 2>() << std::endl;
  std::cout << "replaced_args_list[3]: " <<
      get_value_of_tensor_index<replaced_args_list, 3>() << std::endl;

  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      AUlll{};
  std::iota(AUlll.begin(), AUlll.end(), 0.0);

  auto Aabc = AUlll(ti_A, ti_a, ti_b, ti_c);

  /*auto bc = TensorExpressions::evaluate<ti_b_t, ti_c_t>(Aabc);

  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      All{};

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      All.get(i, j) = 0;
    }
  }

  for (size_t i = 0; i < 3; i++) { // b index value
    for (size_t j = 0; j < 3; j++) { // c index value
      for (size_t m = 0; m < 3; m++) {  // contracted index value (A and a)
        All.get(i, j) += Aabc.get{{(m, m, i, j}});
      }
    }
  }*/
}
