// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function TensorExpressions::evaluate(TensorExpression)

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"

namespace TensorExpressions {

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a Tensor Expression with LHS indices set in the template
 * parameters
 *
 * @tparam LhsIndices the indices on the left hand side of the tensor expression
 * @return Tensor<typename T::type, typename T::symmetry, typename
 * T::index_list>
 */
/*template <typename... LhsIndices, typename T,
          Requires<std::is_base_of<Expression, T>::value> = nullptr>*/

/*template <typename Derived, typename DataType, typename Symm,
          typename IndexList, typename Args = tmpl::list<>,
          typename ReducedArgs = tmpl::list<>>
struct TensorExpression;
/// \endcond

template <typename Derived, typename DataType, typename Symm,
          typename... Indices, template <typename...> class ArgsList,
          typename... Args>
struct TensorExpression<Derived, DataType, Symm, tmpl::list<Indices...>,
                        ArgsList<Args...>>*/

/*template <typename LhsIndexList = tmpl::list<>, typename RhsIndexList =
tmpl::list<>, typename T> Tensor<typename T::type, typename T::symmetry,
typename T::index_list> evaluate_impl(const T& te);

template <typename... LhsIndices, template <typename...> class RhsIndexList,
typename... RhsIndices, typename T> Tensor<typename T::type, typename
T::symmetry, typename T::index_list> evaluate_impl<tmpl::list<LhsIndices...>,
RhsIndexList<RhsIndices...>, T>(const T& te) {*/

/*template <template <typename... LhsIndices> class LhsIndexList, template
<typename... RhsIndices> class RhsIndexList, typename T> Tensor<typename
T::type, typename T::symmetry, typename T::index_list> evaluate_impl(const T&
te) {} constexpr size_t num_indices = sizeof (RhsIndices);

  using lhs_index_list = tmpl::list<LhsIndices...>;
  using rhs_index_list = tmpl::list<RhsIndices...>;
  using rhs_symmetry = typename T::symmetry;

  constexpr std::array<size_t, num_indices> lhs_index_vals =
{{LhsIndices::value...}}; // e.g. 1, 0 for ti_b_t, ti_a_t constexpr
std::array<size_t, num_indices> rhs_index_vals = {{RhsIndices::value...}}; //
e.g. 0, 1 for ti_a_t, ti_b_t

  return Tensor<typename T::type, typename T::symmetry, typename T::index_list>(
      te, tmpl::list<LhsIndices...>{});
}*/

template <size_t Size, class T, T... Ints>
constexpr size_t get_arr_val(const std::integer_sequence<T, Ints...>& /*arr*/,
                             const size_t& i) {
  constexpr std::array<size_t, Size> arr = {Ints...};
  return arr[i];
}

// TODO: ask why I couldn't pass in te - maybe because it is a runtime thing?
template <size_t Size, typename RhsIndexList /*typename TE*/, class T,
          T... Ints>
constexpr std::array<size_t, Size> get_rhs_tensor_index_vals(
    const std::integer_sequence<T, Ints...>& /*indices*/ /*, const TE& te*/) {
  // constexpr std::array<size_t, Size> arr = {Ints...};
  // constexpr std::array<size_t, Size> rhs_tensor_index_values {};

  // using rhs_tensor_indices = typename TE::args_list;
  constexpr std::array<size_t, Size> rhs_tensor_index_values = {
      {tmpl::at_c<RhsIndexList, Ints>::value...}};

  /*for (size_t i = 0; i < Size; i++) {
    constexpr size_t index = arr[i];
    //rhs_tensor_index_values[arr[i]] = tmpl::at_c<rhs_tensor_indices,
  index>::value;
  }*/

  // constexpr std::array<size_t, Size> dummy {};
  return rhs_tensor_index_values;
}

template <size_t Size, class size_t, size_t... Ints>
constexpr std::array<size_t, Size> get_std_arr(
    const std::integer_sequence<size_t, Ints...>& /*indices*/) {
  constexpr std::array<size_t, Size> arr = {Ints...};
  return arr;
}

// TODO: generalize this for rearranging both symmetry and indices?
template <size_t NumberOfIndices>
constexpr std::array<size_t, NumberOfIndices> compute_map(const std::array<
                                                              size_t,
                                                              NumberOfIndices>&
                                                              lhs_index_order,
                                                          const std::array<
                                                              size_t,
                                                              NumberOfIndices>&
                                                              rhs_index_order
                   /*,const std::array<size_t, NumberOfIndices>&
                   lhs_tensor_index*/) noexcept {
  // std::array<size_t, NumberOfIndices> rhs_tensor_index{};
  std::array<size_t, NumberOfIndices> rhs_to_lhs_map{};
  for (size_t i = 0; i < NumberOfIndices; ++i) {
    /*rhs_tensor_index[array_index_of<size_t, NumberOfIndices>(
        rhs_index_order, lhs_index_order[i])] = lhs_tensor_index[i];*/
    // rhs_to_lhs_map[0] = index of rhs_index_order[0] in lhs_index_order
    rhs_to_lhs_map[i] = array_index_of<size_t, NumberOfIndices>(
        lhs_index_order, rhs_index_order[i]);
  }
  return rhs_to_lhs_map;
}

/*template <tmpl::integral_list<std::int32_t, t[Is]...> Symm>
constexpr std::array<int, sizeof...(Ss)> get_symm_as_array() {

}*/

template <size_t NumberOfIndices, typename RhsSymm, class T, T... Ints>
constexpr std::array<std::int32_t, NumberOfIndices> get_lhs_symmetry(
    const std::integer_sequence<T, Ints...>& /*indices*/,
    const std::array<size_t, NumberOfIndices>& rhs_to_lhs_map) {
  // constexpr std::array<std::int32_t, NumberOfIndices> rhs_symmetry =
  // {{RhsSymm::value...}}; // <- doesn't work
  // constexpr std::array<size_t, Size> rhs_tensor_index_values =
  // {{tmpl::at_c<RhsIndexList, Ints>::value...}};
  constexpr std::array<std::int32_t, NumberOfIndices> rhs_symmetry = {
      {tmpl::at_c<RhsSymm, Ints>::value...}};
  std::array<std::int32_t, NumberOfIndices> lhs_symmetry{};
  for (size_t i = 0; i < NumberOfIndices; i++) {
    // rhs_symmetry[0] is the first symmetry #
    // rhs_to_lhs_map[0] = index of rhs_index_order[0] in lhs_index_order
    // so lhs_symmetry[rhs_to_lhs_map[0]] sets lhs[index of rhs_index_order[0]
    // in lhs_index_order] to rhs_symmetry[0] More concisely: the rhs symmetry
    // element is set to be at the index of lhs_symmetry according to mapping
    lhs_symmetry[rhs_to_lhs_map[i]] = rhs_symmetry[i];
  }
  return lhs_symmetry;
}

// here, LhsIndexList and RhsIndexList refer to lists of ti_a_t, ti_b_t, etc.
template <typename IntSequence, typename RhsIndexList, typename LhsIndexList>
struct RhsToLhsIndexMap;

template <size_t... Ints, typename RhsIndexList, /*template <typename...> class
           RhsIndexList, typename... RhsIndices,*/
          typename... LhsIndices>
struct RhsToLhsIndexMap<std::integer_sequence<size_t, Ints...>, RhsIndexList,
                        tmpl::list<LhsIndices...>> {
  static constexpr size_t num_indices = sizeof...(LhsIndices);
  static constexpr std::array<size_t, num_indices> lhs_tensor_index_values = {
      {LhsIndices::value...}};
  static constexpr std::make_integer_sequence<size_t, num_indices> indices{};
  static constexpr std::array<size_t, num_indices> rhs_tensor_index_values =
      get_rhs_tensor_index_vals<num_indices, RhsIndexList>(indices);

  static constexpr std::array<size_t, num_indices> rhs_to_lhs_map =
      compute_map<num_indices>(lhs_tensor_index_values,
                               rhs_tensor_index_values);

  using type = std::integer_sequence<size_t, rhs_to_lhs_map[Ints]...>;
};

// here, LhsIndexList and RhsIndexList refer to lists of
// index types like SpatialIndex<3, UpLo::Lo, Frame::Grid>
template <typename IntSequence, typename RhsSymm, typename RhsToLhsMap>
struct LhsSymm;

template <size_t... Ints, typename RhsSymm, size_t... MapIndices>
struct LhsSymm<std::integer_sequence<size_t, Ints...>, RhsSymm,
               std::integer_sequence<size_t, MapIndices...>> {
  /************ not using ***********/
  // this way works by first generating a std::array for the lhs_symmetry
  static constexpr size_t num_indices = sizeof...(MapIndices);
  static constexpr std::make_integer_sequence<size_t, num_indices> indices{};
  static constexpr std::array<size_t, num_indices> rhs_to_lhs_map = {
      MapIndices...};
  static constexpr std::array<int, num_indices> lhs_symmetry =
      get_lhs_symmetry<num_indices, RhsSymm>(indices, rhs_to_lhs_map);

  using type_not_using =
      tmpl::integral_list<std::int32_t, lhs_symmetry[Ints]...>;
  /*********************************/

  using type = tmpl::list<tmpl::at_c<RhsSymm, MapIndices>...>;
};

// here, LhsIndexList and RhsIndexList refer to lists of
// index types like SpatialIndex<3, UpLo::Lo, Frame::Grid>
template </*typename IntSequence, */ typename RhsIndexList,
          typename RhsToLhsMap>
struct LhsIndexList;

template </*size_t... Ints, template <typename...> class RhsIndexList,
             typename... RhsIndices,*/
          typename RhsIndexList, size_t... MapIndices>
struct LhsIndexList<
    /*std::integer_sequence<size_t, Ints...>, tmpl::list<RhsIndices...>,*/
    RhsIndexList, std::integer_sequence<size_t, MapIndices...>> {
  /*static constexpr cpp20::array<int, sizeof...(Is)> t =
      symmetry(std::array<int, sizeof...(Is)>{{Ss...}});
  using type = tmpl::integral_list<std::int32_t, t[Is]...>;*/

  // using rhs_index_list = tmpl::list<RhsIndices...>;
  using type =
      tmpl::list<tmpl::at_c<RhsIndexList /*rhs_index_list*/, MapIndices>...>;
};

template <typename... LhsIndices, typename T,
          Requires<std::is_base_of<Expression, T>::value> = nullptr>
auto /*Tensor<typename T::type, typename T::symmetry, typename T::index_list>*/
evaluate(const T& te) {
  static_assert(
      sizeof...(LhsIndices) == tmpl::size<typename T::args_list>::value,
      "Must have the same number of indices on the LHS and RHS of a tensor "
      "equation.");
  using rhs = tmpl::transform<tmpl::remove_duplicates<typename T::args_list>,
                              std::decay<tmpl::_1>>;
  static_assert(
      tmpl::equal_members<tmpl::list<std::decay_t<LhsIndices>...>, rhs>::value,
      "All indices on the LHS of a Tensor Expression (that is, those specified "
      "in evaluate<Indices::...>) must be present on the RHS of the expression "
      "as well.");

  // TODO: should there be an assert that you don't have an index repeated?
  //       whether you have repeat with the same or opposite valence, e.g.:
  //           evaluate<ti_a_t, ti_A_t> or evaluate<ti_a_t, ti_a_t>
  //       and other cases, perhaps with more indices even if not repeats, e.g.:
  //           evaluate<ti_a_t, ti_b_t, ti_A_t, ti_c_t>

  // constexpr size_t num_indices = sizeof...(LhsIndices);
  // constexpr std::array<size_t, num_indices> rhs_tensor_indices =

  /*using lhs_index_list = tmpl::list<LhsIndices...>;
  using rhs_tensor_indices = typename T::args_list; // e.g. tmpl::list<ti_a_t,
  ti_b_t> using rhs_symmetry = typename T::symmetry; using rhs_index_list =
  typename T::index_list;

  constexpr std::array<size_t, num_indices> lhs_index_vals =
  {{LhsIndices::value...}}; // e.g. 1, 0 for ti_b_t, ti_a_t constexpr
  std::array<size_t, num_indices> rhs_index_vals = {{T::args_list::value...}};
  // e.g. 0, 1 for ti_a_t, ti_b_t*/

  // obvs can't do
  /*std::array<size_t, num_indices> rhs_to_lhs_map {};
  for (size_t i = 0; i < num_indices; i++) {
    rhs_to_lhs_map[i] = tmpl::index_of<lhs_index_list,
  tmpl::at_c<rhs_tensor_indices, i>>;
  }*/

  using lhs_index_list = tmpl::list<LhsIndices...>;
  using rhs_tensor_indices =
      typename T::args_list;  // e.g. tmpl::list<ti_a_t, ti_b_t>
  using rhs_symmetry = typename T::symmetry;
  using rhs_index_list = typename T::index_list;

  constexpr size_t num_indices = sizeof...(LhsIndices);
  constexpr std::make_integer_sequence<size_t, num_indices>
      indices{};  // [0, 1, ... (num_indices - 1)]
  constexpr std::array<size_t, num_indices> rhs_tensor_index_values =
      get_rhs_tensor_index_vals<num_indices, rhs_tensor_indices>(
          indices /*, te*/);
  constexpr std::array<size_t, num_indices> lhs_tensor_index_values = {
      {LhsIndices::value...}};

  constexpr std::array<size_t, num_indices> rhs_to_lhs_map =
      compute_map<num_indices>(lhs_tensor_index_values,
                               rhs_tensor_index_values);

  // std::integer_sequence<unsigned, 9, 2, 5, 1, 9, 1, 6>{}
  constexpr std::array<std::int32_t, num_indices> lhs_symmetry =
      get_lhs_symmetry<num_indices, rhs_symmetry>(indices, rhs_to_lhs_map);

  using indices_type = std::make_integer_sequence<size_t, num_indices>;
  using map = typename RhsToLhsIndexMap<indices_type, rhs_tensor_indices,
                                        lhs_index_list>::type;
  // map x = 7;
  // std::cout << x << std::endl;
  using lhs_indextype_list = typename LhsIndexList<rhs_index_list, map>::type;

  using lhs_symm_2nd_attempt =
      typename LhsSymm<indices_type, rhs_symmetry, map>::type;

  /*std::array<size_t, num_indices> lhs_to_rhs_map{};
    for (size_t i = 0; i < num_indices; ++i) {
      //rhs_tensor_index[array_index_of<size_t, NumberOfIndices>(
      //    rhs_index_order, lhs_index_order[i])] = lhs_tensor_index[i];
    }*/

  // std::cout << "The RHS tensor index values are : " <<
  // rhs_tensor_index_values << std::endl; constexpr std::array<size_t,
  // num_indices> vals = get_std_arr<num_indices>(indices); constexpr
  // std::array<size_t, num_indices> test {}; constexpr size_t num = vals[0];
  // int j = 0; constexpr size_t num = get_arr_val(indices, j);

  /*std::array<size_t, num_indices> rhs_tensor_index_values {};
  for (size_t i = 0; i < num_indices; i++) {
      //constexpr size_t num = get_arr_val<num_indices>(indices, i); // can't do
  since i is not constexpr
      //test[i] = vals[i];
      //rhs_tensor_index_values[vals[i]] = tmpl::at_c<rhs_tensor_indices,
  vals[i]>::value;
  }*/

  // return evaluate_impl<tmpl::list<LhsIndices...>, typename T::args_list,
  // T>(te);

  /*return Tensor<typename T::type, typename T::symmetry, typename
     T::index_list>( te, tmpl::list<LhsIndices...>{});*/
  return Tensor<typename T::type, lhs_symm_2nd_attempt, lhs_indextype_list>(
      te, tmpl::list<LhsIndices...>{});
}

}  // namespace TensorExpressions
