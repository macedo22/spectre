// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

template <typename List, typename T, T... Ints>
std::array<T, sizeof...(Ints)> get_values_as_array(
    std::integer_sequence<T, Ints...> /*int_seq*/) {
  std::array<T, sizeof...(Ints)> values = {{tmpl::at_c<List, Ints>::value...}};
  return values;
}

template <typename List, size_t Size = tmpl::size<List>::value,
          typename IndexSequence = std::make_index_sequence<Size>>
struct ListVals;

template <typename List, size_t Size, size_t... Ints>
struct ListVals<List, Size, std::index_sequence<Ints...>> {
  // using type = tmpl::list<tmpl::at_c<List, Ints>::value...>;
  static constexpr std::array<size_t, Size> arr = {
      {tmpl::at_c<List, Ints>::value...}};
};

template <typename LhsTensorIndices, typename RhsTnesorIndices>
struct OldRepeatedIndices;

template <typename... LhsTensorIndices, typename... RhsTensorIndices>
struct OldRepeatedIndices<tmpl::list<LhsTensorIndices...>,
                          tmpl::list<RhsTensorIndices...>> {
  // using lhs_repeated_tensorindices = <<10, 1>, <11, 3>>;
  // using rhs_repeated_tensorindices = <<11, 0>, <10, 2>>;
};

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Product",
                  "[DataStructures][Unit]") {
  /*using args_list = tmpl::list<TensorIndex<0>, TensorIndex<4>, TensorIndex<1>,
  TensorIndex<2>, TensorIndex<4>, TensorIndex<8>, TensorIndex<0>>;
  //using repeated_values_returned = repeated_vals<args_list>;
  using args_list_values = ListVals<args_list>::type;
  //using repeated_values_returned = repeated<args_list_values>;
  //constexpr size_t size = tmpl::size<repeated_values_returned>::value;
  std::array<size_t, size> repeated_values_returned_arr =
  get_values_as_array<repeated_values_returned>(
    std::make_index_sequence<size>{});
  std::cout << "\nHere are the repeated tensor index values: " <<
  repeated_values_returned_arr << std::endl;*/

  // using arr = tmpl::list<3, 2, 1>;  // can't do this
}
