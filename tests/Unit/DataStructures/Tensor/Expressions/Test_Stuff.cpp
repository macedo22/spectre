// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

template <size_t I, typename ContractedLhsStructure, size_t Index1,
  size_t Index2, size_t NumContractedIndices, size_t NumUncontractedIndices>
  constexpr std::array<size_t, NumUncontractedIndices>
  fill_contracting_tensor_index() noexcept {
  std::array<size_t, NumUncontractedIndices> contracting_tensor_index{};
  constexpr std::array<size_t, NumContractedIndices>
      lhs_contracted_multi_index =
          ContractedLhsStructure::get_canonical_tensor_index(I);

  for (size_t i = 0; i < Index1; i++) {
    contracting_tensor_index[i] = lhs_contracted_multi_index[i];
  }
  contracting_tensor_index[Index1] = 0;
  for (size_t i = Index1 + 1; i < Index2; i++) {
    contracting_tensor_index[i] = lhs_contracted_multi_index[i - 1];
  }
  contracting_tensor_index[Index2] = 0;
  for (size_t i = Index2 + 1; i < NumUncontractedIndices; i++) {
    contracting_tensor_index[i] = lhs_contracted_multi_index[i - 2];
  }
  return contracting_tensor_index;
}

// template <size_t ToAdd, size_t Index1, size_t Index2, size_t
// NumUncontractedIndices>
template <size_t ToAdd, size_t I, typename ContractedLhsStructure,
          size_t Index1, size_t Index2, size_t NumContractedIndices,
          size_t NumUncontractedIndices>
constexpr std::array<size_t, NumUncontractedIndices>
get_next_tensor_index_to_add(const std::array<size_t, NumUncontractedIndices>
                                 current_contracting_tensor_index) noexcept {
  if constexpr (ToAdd == 0) {
    //(void)current_contracting_tensor_index;
    // return fill_contracting_tensor_index<I, ContractedLhsStructure, Index1,
    // Index2, NumContractedIndices, NumUncontractedIndices>();
    return current_contracting_tensor_index;
  } else {
    // constexpr size_t number_of_indices = sizeof...(MultiIndexInts);
    std::array<size_t, NumUncontractedIndices> next_contracting_tensor_index{};

    for (size_t i = 0; i < Index1; i++) {
      next_contracting_tensor_index[i] = current_contracting_tensor_index[i];
    }
    next_contracting_tensor_index[Index1] =
        current_contracting_tensor_index[Index1] + ToAdd;
    for (size_t i = Index1 + 1; i < Index2; i++) {
      next_contracting_tensor_index[i] = current_contracting_tensor_index[i];
    }
    next_contracting_tensor_index[Index2] =
        current_contracting_tensor_index[Index2] + ToAdd;
    for (size_t i = Index2 + 1; i < NumUncontractedIndices; i++) {
      next_contracting_tensor_index[i] = current_contracting_tensor_index[i];
    }
    return next_contracting_tensor_index;
  }
}

template <size_t I, size_t Dim, typename UncontractedLhsStructure,
          typename ContractedLhsStructure, size_t Index1, size_t Index2,
          size_t NumContractedIndices, size_t NumUncontractedIndices,
          size_t... Ints>
constexpr std::array<size_t, Dim> get_storage_indices_to_sum(
    const std::index_sequence<Ints...>& /*dim_seq*/) noexcept {
  // std::array<size_t, Dim> storage_indices_to_sum{};
  constexpr std::array<size_t, NumUncontractedIndices>
      first_tensor_index_to_sum =
          fill_contracting_tensor_index<I, ContractedLhsStructure, Index1,
                                        Index2, NumContractedIndices,
                                        NumUncontractedIndices>();
  // constexpr size_t first_storage_index_to_sum =
  // UncontractedLhsStructure::get_storage_index(first_tensor_index_to_sum);
  // std::array<size_t, Dim> storage_indices_to_sum[0] =
  // first_storage_index_to_sum;
  ////std::cout << "Here are the first tensor index to sum:  " <<
  ///first_tensor_index_to_sum << std::endl;
  // constexpr std::make_index_sequence<Dim> dim_seq{};
  constexpr std::array<size_t, Dim> base{};
  constexpr std::array<size_t, Dim> storage_indices_to_sum = {
      {UncontractedLhsStructure::get_storage_index(
          get_next_tensor_index_to_add<Ints, I, ContractedLhsStructure, Index1,
                                       Index2, NumContractedIndices,
                                       NumUncontractedIndices>(
              first_tensor_index_to_sum))...}};

  /*for (int i = 1; i < Dim; i++) {

       storage_indices_to_sum[i] = UncontractedLhsStructure::get_storage_index(
           get_next_tensor_index_to_add<Ints, Index1, Index2,
  NumUncontractedIndices>(first_tensor_index_to_sum)
       );
  }*/

  return storage_indices_to_sum;
}

template <size_t NumContractedComponents, size_t Dim,
          typename UncontractedLhsStructure, typename ContractedLhsStructure,
          size_t Index1, size_t Index2, size_t NumContractedIndices,
          size_t NumUncontractedIndices, size_t... Ints>
constexpr std::array<std::array<size_t, Dim>, NumContractedComponents>
get_sum_map(const std::index_sequence<Ints...>& /*index_seq*/) noexcept {
  constexpr std::make_index_sequence<Dim> dim_seq{};
  constexpr std::array<std::array<size_t, Dim>, NumContractedComponents> map = {
      {get_storage_indices_to_sum<Ints, Dim, UncontractedLhsStructure,
                                  ContractedLhsStructure, Index1, Index2,
                                  NumContractedIndices, NumUncontractedIndices>(
          dim_seq)...}};

  return map;
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Stuff",
                  "[DataStructures][Unit]") {
  // Contract first and fourth indices of <2, 1, 1, 1> symmetry rank 4 (upper,
  // lower, lower, lower) tensor to rank 2 tensor and reorder indices
  Tensor<double, Symmetry<2, 1, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rulll{};
  std::iota(Rulll.begin(), Rulll.end(), 0.0);

  const auto RAdba = Rulll(ti_A, ti_d, ti_b, ti_a);
  using RAdba_type = decltype(RAdba);

  const Tensor<double, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      RAdba_contracted_to_bd =
          TensorExpressions::evaluate<ti_b_t, ti_d_t>(RAdba);

  using ContractedLhsStructure =
      Tensor_detail::Structure<Symmetry<1, 1>,
                               SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                               SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>;
  using RhsStructure =
      Tensor_detail::Structure<Symmetry<2, 1, 1, 1>,
                               SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                               SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                               SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                               SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>;
  using UncontractedLhsStructure =
      Tensor_detail::Structure<Symmetry<2, 1, 1, 1>,
                               SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                               SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                               SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                               SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>;
  using LhsTensorIndexList = tmpl::list<ti_b_t, ti_d_t>;
  constexpr size_t I = 0;
  constexpr size_t Index1 = RAdba_type::Index1::value;
  constexpr size_t Index2 = RAdba_type::Index2::value;
  constexpr size_t num_uncontracted_indices = 4;
  constexpr size_t num_contracted_indices = 2;
  constexpr size_t num_contracted_components = ContractedLhsStructure::size();
  constexpr size_t dim = 4;
  // constexpr std::make_index_sequence<num_uncontracted_indices>
  // multi_index_seq{};

  /*constexpr std::array<size_t, dim> first_storage_indices_to_sum =
  get_storage_indices_to_sum<I, dim, UncontractedLhsStructure,
          ContractedLhsStructure,
          Index1, Index2, num_contracted_indices,
          num_uncontracted_indices>();
  std::cout << "Here are the first storage indices to sum:  " <<
  first_storage_indices_to_sum << std::endl;*/

  constexpr std::make_index_sequence<num_contracted_components> map_seq{};
  constexpr std::array<std::array<size_t, dim>, num_contracted_components> map =
      get_sum_map<num_contracted_components, dim, UncontractedLhsStructure,
                  ContractedLhsStructure, Index1, Index2,
                  num_contracted_indices, num_uncontracted_indices>(map_seq);
  std::cout << "Here is the map:  " << map << std::endl;

  for (size_t b = 0; b < 4; b++) {
    for (size_t d = 0; d < 4; d++) {
      const std::array<size_t, 2> tensor_index = {{b, d}};
      const size_t storage_index =
          ContractedLhsStructure::get_storage_index(tensor_index);
      const size_t bd = RAdba_contracted_to_bd.get(b, d);
      const size_t storage_sum = map[storage_index][0] + map[storage_index][1] +
                                 map[storage_index][2] + map[storage_index][3];
      // std::cout << "R(b, d) is : " << bd << ", and the sum of map[" <<
      // storage_index << "] is : " << storage_sum << std::endl;
      CHECK(bd == storage_sum);
    }
  }

  /*constexpr std::array<size_t, num_uncontracted_indices> first_storage_index =
      fill_contracting_tensor_index<I, ContractedLhsStructure, Index1, Index2,
  num_contracted_indices, num_uncontracted_indices>(); std::cout << "Here is
  storage index " << I << "'s first tensor index to sum: " <<
  first_storage_index << std::endl; constexpr std::array<size_t,
  num_uncontracted_indices> second_storage_index =
      fill_contracting_tensor_index<I+1, ContractedLhsStructure, Index1, Index2,
  num_contracted_indices, num_uncontracted_indices>(); std::cout << "Here is
  storage index " << I+1 << "'s first tensor index to sum: " <<
  second_storage_index << std::endl; constexpr std::array<size_t,
  num_uncontracted_indices> third_storage_index =
      fill_contracting_tensor_index<I+2, ContractedLhsStructure, Index1, Index2,
  num_contracted_indices, num_uncontracted_indices>(); std::cout << "Here is
  storage index " << I+2 << "'s first tensor index to sum: " <<
  third_storage_index << std::endl; constexpr std::array<size_t,
  num_uncontracted_indices> fourth_storage_index =
      fill_contracting_tensor_index<I+3, ContractedLhsStructure, Index1, Index2,
  num_contracted_indices, num_uncontracted_indices>(); std::cout << "Here is
  storage index " << I+3 << "'s first tensor index to sum: " <<
  fourth_storage_index << std::endl; constexpr std::array<size_t,
  num_uncontracted_indices> fifth_storage_index =
      fill_contracting_tensor_index<I+4, ContractedLhsStructure, Index1, Index2,
  num_contracted_indices, num_uncontracted_indices>(); std::cout << "Here is
  storage index " << I+4 << "'s first tensor index to sum: " <<
  fifth_storage_index << std::endl; constexpr std::array<size_t,
  num_uncontracted_indices> sixth_storage_index =
      fill_contracting_tensor_index<I+5, ContractedLhsStructure, Index1, Index2,
  num_contracted_indices, num_uncontracted_indices>(); std::cout << "Here is
  storage index " << I+5 << "'s first tensor index to sum: " <<
  sixth_storage_index << std::endl;*/
}
