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

template <typename ReplacedArg1, typename ReplacedArg2, typename T, typename X,
          typename Symm, typename IndexList, typename Args>
using ComputeContractedType = typename ComputeContractedTypeImpl<
    T, X,
    tmpl::erase<
        tmpl::erase<Symm, tmpl::index_of<Args, ReplacedArg2>>,
        tmpl::index_of<tmpl::erase<Args, tmpl::index_of<Args, ReplacedArg2>>,
                       ReplacedArg1>>,
    tmpl::erase<
        tmpl::erase<IndexList, tmpl::index_of<Args, ReplacedArg2>>,
        tmpl::index_of<tmpl::erase<Args, tmpl::index_of<Args, ReplacedArg2>>,
                       ReplacedArg1>>,
    tmpl::erase<
        tmpl::erase<Args, tmpl::index_of<Args, ReplacedArg2>>,
        tmpl::index_of<tmpl::erase<Args, tmpl::index_of<Args, ReplacedArg2>>,
                       ReplacedArg1>>>::type;

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
  static constexpr auto num_uncontracted_tensor_indices =
      tmpl::size<Symm>::value;
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

  // This is a helper function for determining sequential tensor multi-indices
  // that need to be summed in a contraction by setting `ContractedIndexValue`
  // to the values of the indices that were contracted.
  //
  // e.g. Say we contracted R_Aba to L_b. To compute L_1, we need to sum:
  // L_010 + L_111 + L_212. As arrays: [0, 1, 0], [1, 1, 1], and [2, 1, 2]
  //
  // This function is called for
  // `ContractedIndexValue` = [0, dimension of contracted indices). In this
  // example, it means this function is called for the following values for
  // `ContractedIndexValue`: [0, 3). When `ContractedIndexValue` == 0, we return
  // [0, 1, 0], when `ContractedIndexValue` == 1, we return the next index to
  // sum, [1, 1, 1], and finally, when `ContractedIndexValue` == 2, we return
  // the final index to sum, [2, 1, 2].
  //
  // When this function is called for each value of `ContractedIndexValue`, we
  // effectively get a list of tensor indices to sum for a contraction.
  template <size_t ContractedIndexValue, typename ContractedLhsStructure>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<
      size_t, num_uncontracted_tensor_indices>
  get_tensor_index_to_add(const std::array<size_t, num_tensor_indices>&
                              lhs_contracted_multi_index) noexcept {
    std::array<size_t, num_uncontracted_tensor_indices>
        contracting_tensor_index{};

    for (size_t i = 0; i < Index1::value; i++) {
      contracting_tensor_index[i] = lhs_contracted_multi_index[i];
    }
    contracting_tensor_index[Index1::value] = ContractedIndexValue;
    for (size_t i = Index1::value + 1; i < Index2::value; i++) {
      contracting_tensor_index[i] = lhs_contracted_multi_index[i - 1];
    }
    contracting_tensor_index[Index2::value] = ContractedIndexValue;
    for (size_t i = Index2::value + 1; i < num_uncontracted_tensor_indices;
         i++) {
      contracting_tensor_index[i] = lhs_contracted_multi_index[i - 2];
    }
    return contracting_tensor_index;
  }

  // Helper function that gets the uncontracted LHS storage indices that
  // need to be summed to determine the value of the component at storage
  // index I in the contracted LHS expression. In other words, this computes
  // the entry for one storage index (I) in the map computed by `get_sum_map`.
  template <size_t I, typename UncontractedLhsStructure,
            typename ContractedLhsStructure, size_t... Ints>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, CI1::dim>
  get_storage_indices_to_sum(
      const std::index_sequence<Ints...>& /*dim_seq*/) noexcept {
    constexpr std::array<size_t, num_tensor_indices>
        lhs_contracted_multi_index =
            ContractedLhsStructure::get_canonical_tensor_index(I);
    constexpr std::array<size_t, CI1::dim> storage_indices_to_sum = {
        {UncontractedLhsStructure::get_storage_index(
            get_tensor_index_to_add<Ints, ContractedLhsStructure>(
                lhs_contracted_multi_index))...}};

    return storage_indices_to_sum;
  }

  // Computes and returns a mapping between the contracted LHS storage indices
  // and the lists of uncontracted LHS storage indices to sum to determine the
  // values of the components at those contracted LHS storage indices
  //
  // See the comment above the storage index get function below for a more
  // concrete explanation and example of what this mapping is
  template <size_t NumContractedComponents, typename UncontractedLhsStructure,
            typename ContractedLhsStructure, size_t... Ints>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<
      std::array<size_t, CI1::dim>, NumContractedComponents>
  get_sum_map(const std::index_sequence<Ints...>& /*index_seq*/) noexcept {
    constexpr std::make_index_sequence<CI1::dim> dim_seq{};
    constexpr std::array<std::array<size_t, CI1::dim>, NumContractedComponents>
        map = {
            {get_storage_indices_to_sum<Ints, UncontractedLhsStructure,
                                        ContractedLhsStructure>(dim_seq)...}};
    return map;
  }

  // This is a helper that inserts the first contracted index into `LhsIndices`
  template <typename... LhsIndices>
  using uncontracted_lhs_tensorindex_list_helper = tmpl::append<
      tmpl::at_c<tmpl::split_at<tmpl::list<LhsIndices...>, Index1>, 0>,
      tmpl::list<ReplacedArg1>,
      tmpl::at_c<tmpl::split_at<tmpl::list<LhsIndices...>, Index1>, 1>>;

  // This represents the uncontracted LHS indices, where you take the
  // `LhsIndices` and insert the indices that were contracted.
  //
  // e.g. If we contracted RHS R_Abac to LHS L_cb, then this means inserting
  // A and a back in to L_cb in their original spots, which is: L_Acab.
  // Specifically, `uncontracted_lhs_tensorindex_list` would be:
  // `tmpl::list<ti_A_t, ti_c_t, ti_a_t, ti_b_t>`
  template <typename... LhsIndices>
  using uncontracted_lhs_tensorindex_list = tmpl::append<
      tmpl::at_c<
          tmpl::split_at<
              uncontracted_lhs_tensorindex_list_helper<LhsIndices...>, Index2>,
          0>,
      tmpl::list<ReplacedArg2>,
      tmpl::at_c<
          tmpl::split_at<
              uncontracted_lhs_tensorindex_list_helper<LhsIndices...>, Index2>,
          1>>;

  // This returns the contracted component value at the `lhs_storage_index` in
  // the contracted LHS.
  // Iterates over list of uncontracted LHS storage indices of components to
  // sum, gets the components at those indices, and returns their sum
  template <typename UncontractedLhsStructure,
            typename UncontractedLhsTensorIndexList,
            size_t NumContractedComponents, typename T1>
  struct ComputeContraction;

  template <typename UncontractedLhsStructure,
            typename... UncontractedLhsTensorIndices,
            size_t NumContractedComponents, typename T1>
  struct ComputeContraction<UncontractedLhsStructure,
                            tmpl::list<UncontractedLhsTensorIndices...>,
                            NumContractedComponents, T1> {
    static SPECTRE_ALWAYS_INLINE typename T1::type apply(
        const std::array<std::array<size_t, CI1::dim>, NumContractedComponents>&
            map,
        const T1& t1, const size_t& lhs_storage_index) noexcept {
      // This is the first component/value to sum. It assumes that the dimension
      // of the contracted indices is > 0
      //
      // map[lhs_storage_index][0] is the uncontracted LHS storage index of the
      // first component to sum to compute the value of the component at
      // `lhs_storage_index` in the contracted LHS.
      type contraction_sum = t1.template get<UncontractedLhsStructure,
                                             UncontractedLhsTensorIndices...>(
          map[lhs_storage_index][0]);

      // This accumulates the sum of the contraction by getting the values of
      // the other components to sum
      for (size_t i = 1; i < CI1::dim; i++) {
        contraction_sum += t1.template get<UncontractedLhsStructure,
                                           UncontractedLhsTensorIndices...>(
            map[lhs_storage_index][i]);
      }
      return contraction_sum;
    }
  };

  // Storage index get that computes a map from the contracted
  // LHS storage indices to the lists of the uncontracted LHS storage
  // indices of the components to add to carry out the contraction
  // for those contracted storage indices
  //
  // e.g. Say RHS R_Abac was contracted to LHS L_cb and the dim of A/a is 3:
  // L_cb = R_1b1c + R_2b2c + R_3b3c
  //
  // The `lhs_storage_index` passed in represents the storage index of
  // some component of the LHS contracted expression, L_cb. To compute L_cb,
  // we need to `get` the components of the uncontracted tensor to sum,
  // i.e. the 3 components from R_Abac.
  //
  // Since `TensorExpression::get` (storage index) will expect a LHS storage
  // index corresponding to the LHS structure, we need to first construct this
  // uncontracted LHS structure from the `LhsStructure` tparam (the structure
  // of the contracted LHS expression, L_cb). This means keeping the ordering
  // of `LhsIndices` but inserting the indices that were contracted (A, a) into
  // their original positions from the RHS, which is: L_Acab. Then, we get
  // the structure of this "uncontracted LHS". Using this, we can determine
  // the LHS storage indices of components to get to perform the contraction.
  // And from this, we can compute a mapping from the contracted LHS storage
  // indices to lists of the uncontracted LHS storage indices of the components
  // that need to be added to compute the contraction.
  //
  // Example:
  // Let's assume that `lhs_storage_index` = 0 refers to L_11 (c=1, b=1), a
  // component of the contracted LHS expression (L_cb), and we want to get the 3
  // components to add to compute what L_11 is. Let's say that the storage
  // indices for L_1111, L_1212, and L_1313 in the uncontracted LHS expression
  // (L_Acab) are 0, 2, and 5, then: `map[0] = [0, 2, 5]`. Say
  // `lhs_storage_index` = 1 in the contracted expression refers to L_12 (c=1,
  // b=2) and storage indices 1, 4, and 8 refer to L_1112, L_2122, and L_3132 in
  // the uncontracted LHS, then: `map[1] = [1, 4, 8]`. And so forth for all of
  // the contracted LHS storage indices, which yields our map.
  //
  // At runtime, in order to compute some component of L_cb at some
  // `lhs_storage_index`, this function calls ComputeContraction::apply,
  // which accesses `map[lhs_storage_index]` to get the list of
  // uncontracted LHS storage indices of components to add, gets those values,
  // and returns their sum. The `map` is generated by `get_sum_map` and the
  // helper functions that it calls.
  //
  // Motivation:
  // In order to leverage the benefits of TensorExpression's
  // storage index get, contractions need to also use a storage index get.
  // Currently, the multi-index get for contractions (above in this file)
  // determines which tensor indices to get to add at runtime, and it also calls
  // ComputeContractionImpl, which calls TensorExpression's multi-index get.
  // This means that in addition to computing which tensor indices to add at
  // runtime, it also means that we aren't leveraging the O(1) array lookup of
  // the TensorExpression storage index get when we go to access a component of
  // a tensor (because ComputeContractionImpl calls the multi-index get).
  // We could just convert the multi-index in ComputeContractionImpl to the
  // storage index and then call the storage index get, but that defeats the
  // purpose/benefit of the TensorExpression storage index get.
  //
  // In addition, doing this mapping work up front provides the same benefit
  // as TensorExpression's storage index get: we are able to simply do an
  // array lookup of each component to add in the contraction.
  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE type get(const size_t lhs_storage_index) const {
    // number of components in the contracted LHS
    constexpr size_t num_contracted_components = LhsStructure::size();

    // structure of the uncontracted LHS (LHS with contracted indices inserted)
    using UncontractedLhsStructure = typename LhsTensorSymmAndIndices<
        ArgsList, uncontracted_lhs_tensorindex_list<LhsIndices...>, Symm,
        IndexList>::structure;

    constexpr std::make_index_sequence<num_contracted_components> map_seq{};

    // map from contracted LHS storage indices to lists of uncontracted LHS
    // storage indices of components to sum for contraction
    constexpr std::array<std::array<size_t, CI1::dim>,
                         num_contracted_components>
        map = get_sum_map<num_contracted_components, UncontractedLhsStructure,
                          LhsStructure>(map_seq);

    // This returns the contracted component value at the `lhs_storage_index` in
    // the contracted LHS.
    return ComputeContraction<UncontractedLhsStructure,
                              uncontracted_lhs_tensorindex_list<LhsIndices...>,
                              num_contracted_components,
                              decltype(t_)>::apply(map, t_, lhs_storage_index);
  }

 private:
  const std::conditional_t<std::is_base_of<Expression, T>::value, T,
                           TensorExpression<T, X, Symm, IndexList, ArgsList>>
      t_;
};

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename ReplacedArg1, typename ReplacedArg2,
          typename T, typename X, typename Symm, typename IndexList,
          typename Args>
SPECTRE_ALWAYS_INLINE auto contract(
    const TensorExpression<T, X, Symm, IndexList, Args>& t) {
  return TensorContract<ReplacedArg1, ReplacedArg2, T, X, Symm, IndexList,
                        Args>(~t);
}

namespace detail {
// Helper struct to allow contractions by using repeated indices in operator()
// calls to tensor.
template <template <typename> class TE, typename ReplacedArgList, typename I,
          typename TotalContracted>
struct fully_contract_helper {
  using lower_tensorindex = ti_contracted_t<I::value, UpLo::Lo>;
  using upper_tensorindex = TensorIndex<lower_tensorindex::value + 1, UpLo::Up>;
  using ReplacedArg1 = tmpl::conditional_t<
      (tmpl::index_of<ReplacedArgList, lower_tensorindex>::value <
       tmpl::index_of<ReplacedArgList, upper_tensorindex>::value),
      lower_tensorindex, upper_tensorindex>;
  using ReplacedArg2 = tmpl::conditional_t<
      (tmpl::index_of<ReplacedArgList, lower_tensorindex>::value <
       tmpl::index_of<ReplacedArgList, upper_tensorindex>::value),
      upper_tensorindex, lower_tensorindex>;

  template <typename T>
  SPECTRE_ALWAYS_INLINE static constexpr auto apply(const T& t)
      -> decltype(contract<ReplacedArg1, ReplacedArg2>(
          fully_contract_helper<TE, ReplacedArgList, tmpl::size_t<I::value + 1>,
                                TotalContracted>::apply(t))) {
    return contract<ReplacedArg1, ReplacedArg2>(
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
  using lower_tensorindex = ti_contracted_t<I::value, UpLo::Lo>;
  using upper_tensorindex = TensorIndex<lower_tensorindex::value + 1, UpLo::Up>;
  using ReplacedArg1 = tmpl::conditional_t<
      (tmpl::index_of<ReplacedArgList, lower_tensorindex>::value <
       tmpl::index_of<ReplacedArgList, upper_tensorindex>::value),
      lower_tensorindex, upper_tensorindex>;
  using ReplacedArg2 = tmpl::conditional_t<
      (tmpl::index_of<ReplacedArgList, lower_tensorindex>::value <
       tmpl::index_of<ReplacedArgList, upper_tensorindex>::value),
      upper_tensorindex, lower_tensorindex>;

  template <typename T>
  SPECTRE_ALWAYS_INLINE static constexpr auto apply(const T& t) -> decltype(
      contract<ReplacedArg1, ReplacedArg2>(TE<ReplacedArgList>(t))) {
    return contract<ReplacedArg1, ReplacedArg2>(TE<ReplacedArgList>(t));
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
