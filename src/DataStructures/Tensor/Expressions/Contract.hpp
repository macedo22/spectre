// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Expression Templates for contracting tensor indices on a single
/// Tensor

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Requires.hpp"

/*!
 * \ingroup TensorExpressionsGroup
 * Holds all possible TensorExpressions currently implemented
 */
namespace TensorExpressions {

namespace detail {

template <typename I1, typename I2>
using indices_contractible =
    std::integral_constant<
        bool,
        I1::dim == I2::dim and I1::ul != I2::ul and
            std::is_same_v<typename I1::Frame, typename I2::Frame> and
            I1::index_type == I2::index_type>;

template <typename T, typename X, typename SymmList, typename IndexList,
          typename Args>
struct ComputeContractedTypeImpl;

template <typename T, typename X, template <typename...> class SymmList,
          typename IndexList, typename Args, typename... Symm>
struct ComputeContractedTypeImpl<T, X, SymmList<Symm...>, IndexList, Args> {
  using type =
      TensorExpression<T, X, Symmetry<Symm::value...>, IndexList, Args>;
};

/*template <typename ReplacedArg1, typename ReplacedArg2, typename T, typename
X, typename Symm, typename IndexList, typename Args> using ComputeContractedType
= typename ComputeContractedTypeImpl< T, X, tmpl::erase<tmpl::erase<Symm,
tmpl::index_of< Args, TensorIndex<ReplacedArg2::value>>>, tmpl::index_of<Args,
TensorIndex<ReplacedArg1::value>>>, tmpl::erase< tmpl::erase<IndexList,
                    tmpl::index_of<Args, TensorIndex<ReplacedArg2::value>>>,
        tmpl::index_of<Args, TensorIndex<ReplacedArg1::value>>>,
    tmpl::erase<tmpl::erase<Args, tmpl::index_of<
                                      Args, TensorIndex<ReplacedArg2::value>>>,
                tmpl::index_of<Args,
TensorIndex<ReplacedArg1::value>>>>::type;*/

template <typename ReplacedArg1, typename ReplacedArg2, typename T, typename X,
          typename Symm, typename IndexList, typename Args>
using ComputeContractedType = typename ComputeContractedTypeImpl<
    T, X,
    tmpl::erase<tmpl::erase<Symm, tmpl::index_of<Args, ReplacedArg2>>,
                tmpl::index_of<Args, ReplacedArg1>>,
    tmpl::erase<tmpl::erase<IndexList, tmpl::index_of<Args, ReplacedArg2>>,
                tmpl::index_of<Args, ReplacedArg1>>,
    tmpl::erase<tmpl::erase<Args, tmpl::index_of<Args, ReplacedArg2>>,
                tmpl::index_of<Args, ReplacedArg1>>>::type;

template <int I, typename Index1, typename Index2>
struct ComputeContractionImpl {
  template <typename... LhsIndices, typename T, typename S>
  static SPECTRE_ALWAYS_INLINE typename T::type apply(S tensor_index,
                                                      const T& t1) {
    tensor_index[Index1::value] = I;
    tensor_index[Index2::value] = I;
    return t1.template get<LhsIndices...>(tensor_index) +
           ComputeContractionImpl<I - 1, Index1, Index2>::template apply<
               LhsIndices...>(tensor_index, t1);
  }
};

template <typename Index1, typename Index2>
struct ComputeContractionImpl<0, Index1, Index2> {
  template <typename... LhsIndices, typename T, typename S>
  static SPECTRE_ALWAYS_INLINE typename T::type apply(S tensor_index,
                                                      const T& t1) {
    tensor_index[Index1::value] = 0;
    tensor_index[Index2::value] = 0;
    return t1.template get<LhsIndices...>(tensor_index);
  }
};

/*template <typename LhsTensorIndexList, typename LhsNumComponents, typename
RhsTensorIndexList, typename RhsTensorIndexTypeList, typename RhsSymmetry,
typename RhsDim, size_t RhsNumComponents, typename LhsIndexSequence =
std::make_index_sequence<LhsNumIndices>, typename RhsIndexSequence =
std::make_index_sequence<RhsNumComponents>> struct LhsStorageToIndexMap;

template <typename... LhsTensorIndices, typename LhsNumComponents, typename...
RhsTensorIndices, typename RhsTensorTypeIndexList, typename RhsSymmetry,
typename RhsDim, size_t RhsNumComponents, size_t... LhsInts, size_t... RhsInts>
struct LhsStorageToIndexMap<tmpl::list<LhsTensorIndices...>, LhsNumComponents,
tmpl::list<RhsTensorIndices...>, RhsTensorTypeIndexList, RhsSymmetry, RhsDim,
RhsNumComponents, std::index_sequence<LhsInts...>,
std::index_sequence<RhsInts...>>{ SPECTRE_ALWAYS_INLINE static constexpr
std::array<std::array<size_t, RhsDim>, LhsNumComponents> {
//constexpr size_t lhs_num_components = sizeof...(LhsTensorIndices);
//constexpr size_t rhs_num_indices = sizeof...(RhsTensorIndices);
 apply() noexcept {
     //constexpr size_t num_components = CI1::dim;
     std::array<std::array<size_t, RhsDim>, LhsNumComponents>
         lhs_storage_sum_map{};
     //using rhs_structure = uncontracted_structure;
     using rhs_tensorindex_list = tmpl::list<RhsTensorIndices...>;
     using lhs_tensorindex_list = //somehow fill this
     //using rhs_symmetry = RhsSymmetry;
     //using rhs_tensorindextype_list = tmpl::list<>
     auto lhs_storage_to_tensor_indices =
lhs_structure::storage_to_tensor_index();
}
};*/
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename ReplacedArg1, typename ReplacedArg2, typename T, typename X,
          typename Symm, typename IndexList, typename ArgsList>
struct TensorContract
    : public TensorExpression<TensorContract<ReplacedArg1, ReplacedArg2, T, X,
                                             Symm, IndexList, ArgsList>,
                              X,
                              typename detail::ComputeContractedType<
                                  ReplacedArg1, ReplacedArg2, T, X, Symm,
                                  IndexList, ArgsList>::symmetry,
                              typename detail::ComputeContractedType<
                                  ReplacedArg1, ReplacedArg2, T, X, Symm,
                                  IndexList, ArgsList>::index_list,
                              typename detail::ComputeContractedType<
                                  ReplacedArg1, ReplacedArg2, T, X, Symm,
                                  IndexList, ArgsList>::args_list> {
  /*using Index1 = tmpl::size_t<
      tmpl::index_of<ArgsList, TensorIndex<ReplacedArg1::value>>::value>;
  using Index2 = tmpl::size_t<
      tmpl::index_of<ArgsList, TensorIndex<ReplacedArg2::value>>::value>;*/
  using Index1 = tmpl::size_t<tmpl::index_of<ArgsList, ReplacedArg1>::value>;
  using Index2 = tmpl::size_t<tmpl::index_of<ArgsList, ReplacedArg2>::value>;
  using CI1 = tmpl::at<IndexList, Index1>;
  using CI2 = tmpl::at<IndexList, Index2>;
  static_assert(tmpl::size<Symm>::value > 1 and
                    tmpl::size<IndexList>::value > 1,
                "Cannot contract indices on a Tensor with rank less than 2");
  static_assert(detail::indices_contractible<CI1, CI2>::value,
                "Cannot contract the requested indices.");

  using new_type = detail::ComputeContractedType<ReplacedArg1, ReplacedArg2, T,
                                                 X, Symm, IndexList, ArgsList>;

  using type = X;
  using symmetry = typename new_type::symmetry;
  using index_list = typename new_type::index_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  using args_list = typename new_type::args_list;
  using structure = Tensor_detail::Structure<symmetry, index_list>;

  explicit TensorContract(
      const TensorExpression<T, X, Symm, IndexList, ArgsList>& t)
      : t_(~t) {}

  template <size_t I, size_t Rank>
  SPECTRE_ALWAYS_INLINE void fill_contracting_tensor_index(
      std::array<size_t, Rank>& tensor_index_in,
      const std::array<size_t, num_tensor_indices>& tensor_index) const {
    if constexpr (I <= Index1::value) {
      // 10000 is for the slot that will be set later. Easy to debug.
      tensor_index_in[I] = I == Index1::value ? 10000 : tensor_index[I];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I > Index1::value and I <= Index2::value and
                         I < Rank - 1) {
      // tensor_index is Rank - 2 since it shouldn't be called for Rank 2 case
      // 20000 is for the slot that will be set later. Easy to debug.
      tensor_index_in[I] = I == Index2::value ? 20000 : tensor_index[I - 1];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I > Index2::value and I < Rank - 1) {
      // Left as Rank - 2 since it should never be called for the Rank 2 case
      tensor_index_in[I] = tensor_index[I - 2];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else {
      tensor_index_in[I] = I == Index2::value ? 20000 : tensor_index[I - 2];
    }
  }

  template <typename... LhsIndices, typename U>
  SPECTRE_ALWAYS_INLINE type
  get(const std::array<U, num_tensor_indices>& new_tensor_index) const {
    // new_tensor_index is the one with _fewer_ components, ie post-contraction
    std::array<size_t, tmpl::size<Symm>::value> tensor_index;
    // Manually unrolled for loops to compute the tensor_index from the
    // new_tensor_index
    fill_contracting_tensor_index<0>(tensor_index, new_tensor_index);
    return detail::ComputeContractionImpl<CI1::dim - 1, Index1, Index2>::
        template apply<LhsIndices...>(tensor_index, t_);
  }

  /*template<typename LhsNumIndices, typename LhsTensorIndexList, typename
  RhsTensorIndexList> SPECTRE_ALWAYS_INLINE static constexpr
  std::array<std::array<size_t, CI1::dim>, sizeof... (LhsInts)>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, CI1::dim>
  compute_lhs_storage_indices(const std::array<size_t, LhsNumIndices>&
  lhs_tensor_index) {

  }*/

  /*template<size_t I, typename LhsTensorIndexList, typename RhsTensorIndexList,
 typename UnContractedLhsTensorIndexList = tmpl::list<>> struct
 LhsUncontractedIndices;

  template<size_t I, typename... LhsTensorIndices, typename... RhsTensorIndices,
 typename UnContractedLhsTensorIndexList> struct LhsUncontractedIndices<I,
 tmpl::list<LhsTensorIndices...>, tmpl::list<RhsTensorIndices...>,
 UnContractedLhsTensorIndexList> { static constexpr size_t lhs_num_indices =
 tmpl::size<Symm>::value - 1; using lhs_tensorindex_list =
 tmpl::list<LhsTensorIndices...>; using rhs_tensorindex_list =
 tmpl::list<RhsTensorIndices...>;
    //using type = tmpl::list<std::conditional_t<I == Index1, CI1,
 std::conditional_t<I == Index1, CI2, ?>>::value>; using type = brigand::insert<
        UnContractedLhsTensorIndexList,
        LhsUncontractedIndices<I+1, tmpl::list<LhsTensorIndices...>,
 tmpl::list<RhsTensorIndices...>, tmpl::list<>>::type>
     >;
  };

  template<size_t I, typename... LhsTensorIndices, typename... RhsTensorIndices,
 typename UnContractedLhsTensorIndex = tmpl::list<>> struct
 LhsUncontractedIndices<tmpl::size<Symm>::value - 1,
 tmpl::list<LhsTensorIndices...>, tmpl::list<RhsTensorIndices...>> { static
 constexpr size_t rhs_num_indices = tmpl::size<Symm>::value; using
 lhs_tensorindex_list = tmpl::list<LhsTensorIndices...>; using
 rhs_tensorindex_list = tmpl::list<RhsTensorIndices...>;
    //using type = tmpl::list<std::conditional_t<I == Index1, CI1,
 std::conditional_t<I == Index1, CI2, ?>>::value>; using type = brigand::insert<
        UnContractedLhsTensorIndex,
        std::conditional_t<
           I == Index1,
           CI1,
           std::conditional_t<
             I == Index2,
             CI2,
             tmpl::at_c<lhs_tensorindex_list, I>
           >
        >
     >;

  };

 template <size_t I, typename LhsStructure, typename... LhsTensorIndices,
 size_t... MapInts> SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t,
 tmpl::size<Symm>::value> fill_contracting_tensor_index( const
 std::index_sequence<MapInts...>& map_seq) noexcept { constexpr size_t
 number_of_indices = tmpl::size<Symm>::value; std::array<size_t,
 number_of_indices> contracting_tensor_index{}; constexpr std::array<size_t,
 number_of_indices> lhs_multi_index =
 LhsStructure::get_canonical_tensor_index(I); contracting_tensor_index[Index1] =
 0; contracting_tensor_index[Index2] = 0;

   for (int i = 0; i < Index1; i++) {
     contracting_tensor_index[i] = lhs_multi_index[i];
   }
   contracting_tensor_index[Index1] = 0;
   for (int i = Index1 + 1; i < Index2; i++) {
     contracting_tensor_index[i] = lhs_multi_index[i - 1];
   }
   contracting_tensor_index[Index2] = 0;
   for (int i = Index2 + 1; i < number_of_indices; i++) {
     contracting_tensor_index[i] = lhs_multi_index[i - 2];
   }
   return rhs_tensor_multi_index;
 }

  template<size_t NumComponents, typename LhsStructure, typename...
 LhsTensorIndices, typename... LhsInts, typename... RhsInts, typename...
 MapInts> SPECTRE_ALWAYS_INLINE static constexpr std::array<std::array<size_t,
                                                  CI1::dim>,
                                                  sizeof... (LhsInts)>
  compute_lhs_storage_to_tensor_map(const std::integer_sequence<T, LhsInts...>&
 rhs_seq, const std::integer_sequence<T, RhsInts...>& lhs_seq, const
 std::integer_sequence<T, MapInts...>& map_seq) noexcept { constexpr size_t
 lhs_num_components = sizeof... (LhsInts); constexpr size_t lhs_num_indices =
 tmpl::size<LhsStructure::index_list>::value; constexpr size_t rhs_dim =
 CI1::dim; static constexpr size_t rhs_num_indices = tmpl::size<Symm>::value;
    constexpr std::make_index_sequence<NumComponents> map_seq{};
    //std::array<std::array<size_t, rhs_dim>, lhs_num_components>
    //    lhs_storage_sum_map{};
    constexpr auto lhs_storage_to_tensor_indices =
 LhsStructure::storage_to_tensor_index();
    //for (int i = 0; i < lhs_num_components; i++) {
    //  //const std::conditional_t<std::is_base_of<Expression, T>::value, T,
    //  //                    TensorExpression<T, X, Symm, IndexList, ArgsList>>
    //  lhs_storage_sum_map[i] = compute_lhs_storage_indices<lhs_num_indices,
 tmpl::list<LhsTensorIndices>, IndexList>(lhs_storage_to_tensor_indices[i]);
    //}
    //using rhs_structure = uncontracted_structure;
    using rhs_tensorindex_list = tmpl::list<RhsTensorIndices...>;
    //using lhs_tensorindex_list = somehow fill this
    constexpr std::array<std::array<size_t, rhs_dim>, lhs_num_components>
        lhs_storage_sum_map = {{fill_contracting_tensor_index<MapInts,
 LhsTensorIndices...>(multi_index_seq)...}};
    //using rhs_symmetry = RhsSymmetry;
    //using rhs_tensorindextype_list = tmpl::list<>
    auto lhs_storage_to_tensor_indices =
 lhs_structure::storage_to_tensor_index();
 }*/

  // TODO: find a way to propagate using the storage index
  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE type get(const size_t lhs_storage_index) const {
    // maybe make a map from LHS storage index to the new array oh LHS
    // indices to add
    // maybe use LhsSymmAndIndices to get the expanded LHS symm and indices
    // to and may need to update LhsStructure and LhsIndices for recursive
    // calls to get
    // maybe need to use an index sequence for figuring out the storage
    // indices that need to be added

    // Possible implementation:
    //
    // constexpr std::array<std::array<size_t, num_tensor_indices>,
    // num_components>
    //     lhs_storage_indices_to_add =
    //

    /*constexpr size_t num_contracted_components = LhsStructure::size();
    constexpr std::make_index_sequence<num_contracted_components> map_seq{};
    constexpr std::array<std::array<size_t, C1::dim>, num_contracted_components>
    map = get_sum_map<num_contracted_components, C1::dim,
    UncontractedLhsStructure, LhsStructure, Index1, Index2,
                  num_tensor_indices, tmpl::size<Symm>::value>(map_seq);*/

    const std::array<size_t, num_tensor_indices>& new_tensor_index =
        LhsStructure::template get_canonical_tensor_index<num_tensor_indices>(
            lhs_storage_index);
    return get<LhsStructure, LhsIndices...>(new_tensor_index);
  }

 private:
  const std::conditional_t<std::is_base_of<Expression, T>::value, T,
                           TensorExpression<T, X, Symm, IndexList, ArgsList>>
      t_;
};

/*!
 * \ingroup TensorExpressionsGroup
 */
template </*int*/ typename ReplacedArg1, /*int*/ typename ReplacedArg2,
          typename T, typename X, typename Symm, typename IndexList,
          typename Args>
SPECTRE_ALWAYS_INLINE auto contract(
    const TensorExpression<T, X, Symm, IndexList, Args>& t) {
  /*return TensorContract<tmpl::size_t<ReplacedArg1>,
                        tmpl::size_t<ReplacedArg2>, T, X, Symm, IndexList,
                        Args>(~t);*/
  return TensorContract<ReplacedArg1, ReplacedArg2, T, X, Symm, IndexList,
                        Args>(~t);
}

namespace detail {
// Helper struct to allow contractions by using repeated indices in operator()
// calls to tensor.
template <template <typename> class TE, typename ReplacedArgList, typename I,
          typename TotalContracted>
struct fully_contract_helper {
  template <typename T>
  SPECTRE_ALWAYS_INLINE static constexpr auto apply(const T& t) -> decltype(
      contract<ti_contracted_t<I::value /*, UpLo::Lo*/> /*::value*/,
               ti_contracted_t<I::value + 1 /*, UpLo::Up*/> /*::value*/>(
          fully_contract_helper<TE, ReplacedArgList, tmpl::size_t<I::value + 1>,
                                TotalContracted>::apply(t))) {
    return contract<ti_contracted_t<I::value /*, UpLo::Lo*/> /*::value*/,
                    ti_contracted_t<I::value + 1 /*, UpLo::Up*/> /*::value*/>(
        fully_contract_helper<TE, ReplacedArgList, tmpl::size_t<I::value + 1>,
                              TotalContracted>::apply(t));
  }
};

template <template <typename> class TE, typename ReplacedArgList,
          typename TotalContracted>
struct fully_contract_helper<TE, ReplacedArgList,
                             tmpl::size_t<TotalContracted::value - 1>,
                             TotalContracted> {
  using I = tmpl::size_t<2 * (TotalContracted::value - 1)>;
  template <typename T>
  SPECTRE_ALWAYS_INLINE static constexpr auto apply(const T& t) -> decltype(
      contract<ti_contracted_t<I::value /*, UpLo::Lo*/> /*::value*/,
               ti_contracted_t<I::value + 1 /*, UpLo::Up*/> /*::value*/>(
          TE<ReplacedArgList>(t))) {
    return contract<ti_contracted_t<I::value /*, UpLo::Lo*/> /*::value*/,
                    ti_contracted_t<I::value + 1 /*, UpLo::Up*/> /*::value*/>(
        TE<ReplacedArgList>(t));
  }
};
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Represents a fully contracted Tensor
 */
template <template <typename> class TE, typename ReplacedArgList, typename I,
          typename TotalContracted>
using fully_contracted =
    detail::fully_contract_helper<TE, ReplacedArgList, I, TotalContracted>;
}  // namespace TensorExpressions
