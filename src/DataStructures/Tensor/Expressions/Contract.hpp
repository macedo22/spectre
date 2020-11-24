// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Expression Templates for contracting tensor indices on a single
/// Tensor

#pragma once

#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
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
using indices_contractible = std::integral_constant<
    bool, I1::dim == I2::dim and I1::ul != I2::ul and
              std::is_same_v<typename I1::Frame, typename I2::Frame> and
              I1::index_type == I2::index_type>;

template <typename T, typename X, typename SymmList, typename IndexList,
          typename TensorIndexList>
struct ContractedTypeImpl;

template <typename T, typename X, template <typename...> class SymmList,
          typename IndexList, typename TensorIndexList, typename... Symm>
struct ContractedTypeImpl<T, X, SymmList<Symm...>, IndexList, TensorIndexList> {
  using type = TensorExpression<T, X, Symmetry<Symm::value...>, IndexList,
                                TensorIndexList>;
};

template <size_t FirstContractedIndexPos, size_t SecondContractedIndexPos,
          typename T, typename X, typename Symm, typename IndexList,
          typename TensorIndexList>
struct ContractedType {
  static_assert(FirstContractedIndexPos < SecondContractedIndexPos,
                "The position of the first provided index to contract must be "
                "less than the position of the second index to contract.");
  using contracted_symmetry =
      tmpl::erase<tmpl::erase<Symm, tmpl::size_t<SecondContractedIndexPos>>,
                  tmpl::size_t<FirstContractedIndexPos>>;
  using contracted_index_list = tmpl::erase<
      tmpl::erase<IndexList, tmpl::size_t<SecondContractedIndexPos>>,
      tmpl::size_t<FirstContractedIndexPos>>;
  using contracted_tensorindex_list = tmpl::erase<
      tmpl::erase<TensorIndexList, tmpl::size_t<SecondContractedIndexPos>>,
      tmpl::size_t<FirstContractedIndexPos>>;
  using type = typename ContractedTypeImpl<T, X, contracted_symmetry,
                                           contracted_index_list,
                                           contracted_tensorindex_list>::type;
};

template <size_t I, size_t Index1, size_t Index2, typename... LhsIndices,
          typename T, typename S>
static SPECTRE_ALWAYS_INLINE decltype(auto) compute_contraction(S tensor_index,
                                                                const T& t1) {
  if constexpr (I == 0) {
    tensor_index[Index1] = 0;
    tensor_index[Index2] = 0;
    return t1.template get<LhsIndices...>(tensor_index);
  } else {
    tensor_index[Index1] = I;
    tensor_index[Index2] = I;
    return t1.template get<LhsIndices...>(tensor_index) +
           compute_contraction<I - 1, Index1, Index2, LhsIndices...>(
               tensor_index, t1);
  }
}
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 */
template <size_t FirstContractedIndexPos, size_t SecondContractedIndexPos,
          typename T, typename X, typename Symm, typename IndexList,
          typename ArgsList>
struct TensorContract
    : public TensorExpression<
          TensorContract<FirstContractedIndexPos, SecondContractedIndexPos, T,
                         X, Symm, IndexList, ArgsList>,
          X,
          typename detail::ContractedType<FirstContractedIndexPos,
                                          SecondContractedIndexPos, T, X, Symm,
                                          IndexList, ArgsList>::type::symmetry,
          typename detail::ContractedType<
              FirstContractedIndexPos, SecondContractedIndexPos, T, X, Symm,
              IndexList, ArgsList>::type::index_list,
          typename detail::ContractedType<
              FirstContractedIndexPos, SecondContractedIndexPos, T, X, Symm,
              IndexList, ArgsList>::type::args_list> {
  // First and second \ref SpacetimeIndex "TensorIndexType"s to contract.
  // "first" and "second" here refer to the position of the indices to contract
  // in the list of indices, with "first" denoting leftmost
  //
  // e.g. `R(ti_A, ti_b, ti_a)` :
  // - `first_contracted_index` refers to the
  //   \ref SpacetimeIndex "TensorIndexType" refered to by `ti_A`
  // - `second_contracted_index` refers to the
  //   \ref SpacetimeIndex "TensorIndexType" refered to by `ti_a`
  using first_contracted_index = tmpl::at_c<IndexList, FirstContractedIndexPos>;
  using second_contracted_index =
      tmpl::at_c<IndexList, SecondContractedIndexPos>;
  static_assert(tmpl::size<Symm>::value > 1 and
                    tmpl::size<IndexList>::value > 1,
                "Cannot contract indices on a Tensor with rank less than 2");
  static_assert(detail::indices_contractible<first_contracted_index,
                                             second_contracted_index>::value,
                "Cannot contract the requested indices.");

  using new_type =
      typename detail::ContractedType<FirstContractedIndexPos,
                                      SecondContractedIndexPos, T, X, Symm,
                                      IndexList, ArgsList>::type;

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
    if constexpr (I < FirstContractedIndexPos) {
      tensor_index_in[I] = tensor_index[I];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I == FirstContractedIndexPos) {
      // 10000 is for the slot that will be set later. Easy to debug.
      tensor_index_in[I] = 10000;
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I > FirstContractedIndexPos and
                         I <= SecondContractedIndexPos and I < Rank - 1) {
      // tensor_index is Rank - 2 since it shouldn't be called for Rank 2 case
      // 20000 is for the slot that will be set later. Easy to debug.
      tensor_index_in[I] =
          I == SecondContractedIndexPos ? 20000 : tensor_index[I - 1];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I > SecondContractedIndexPos and I < Rank - 1) {
      // Left as Rank - 2 since it should never be called for the Rank 2 case
      tensor_index_in[I] = tensor_index[I - 2];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I == SecondContractedIndexPos) {
      tensor_index_in[I] = 20000;
    } else {
      tensor_index_in[I] = tensor_index[I - 2];
    }
  }

  template <typename... LhsIndices, typename U>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<U, num_tensor_indices>& new_tensor_index) const {
    // new_tensor_index is the one with _fewer_ components, ie post-contraction
    std::array<size_t, tmpl::size<Symm>::value> tensor_index{};
    // Manually unrolled for loops to compute the tensor_index from the
    // new_tensor_index
    fill_contracting_tensor_index<0>(tensor_index, new_tensor_index);
    return detail::compute_contraction<first_contracted_index::dim - 1,
                                       FirstContractedIndexPos,
                                       SecondContractedIndexPos, LhsIndices...>(
        tensor_index, t_);
  }

  /// \brief return a tensor multi-index of the uncontracted LHS to be summed in
  /// a contraction
  ///
  /// \details
  /// Given a RHS tensor to be contracted, the uncontracted LHS represents the
  /// uncontracted RHS tensor arranged with the LHS's generic index order. The
  /// contracted LHS represents the result of contracting this uncontracted
  /// LHS. For example, if we have RHS tensor \f${R^{a}}_{abc}\f$ and we want to
  /// contract it to the LHS tensor \f$L_{cb}\f$, then \f$L_{cb}\f$ represents
  /// the contracted LHS, while \f${L^{a}}_{acb}\f$ represents the uncontracted
  /// LHS. Note that the relative ordering of the LHS generic indices \f$c\f$
  /// and \f$b\f$ in the contracted LHS is preserved in the uncontracted LHS.
  ///
  /// To compute a contraction, we need to get all the uncontracted LHS
  /// multi-indices to sum. In the example above, this means that in order to
  /// compute \f$L_{cb}\f$ for some \f$c\f$ and \f$b\f$, we need to sum the
  /// components \f${L^{a}}_{acb}\f$ for all values of \f$a\f$. This function
  /// takes a concrete contracted LHS multi-index as input, representing the
  /// multi-index of a component of the contracted LHS. For example, if
  /// `lhs_contracted_multi_index == [1, 2]`, this represents \f$L_{12}\f$.
  /// In this case, we need to sum \f${L^{a}}_{a12}\f$ for all values of
  /// \f$a\f$. `ContractedIndexValue` represents on such concrete value that is
  /// filled in for \f$a\f$. In this way, what is constructed and returned is
  /// one such concrete uncontracted LHS multi-index to be summed as part of
  /// contracting a pair of indices.
  ///
  /// \tparam ContractedIndexValue concrete value inserted for the indices to
  /// contract
  /// \param lhs_contracted_multi_index a tensor multi-index of the contracted
  /// LHS
  /// \return a tensor multi-index of the uncontracted LHS to be summed in a
  /// contraction
  template <size_t ContractedIndexValue>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<
      size_t, num_uncontracted_tensor_indices>
  get_tensor_index_to_sum(const std::array<size_t, num_tensor_indices>&
                              lhs_contracted_multi_index) noexcept {
    std::array<size_t, num_uncontracted_tensor_indices>
        contracting_tensor_index{};

    for (size_t i = 0; i < FirstContractedIndexPos; i++) {
      contracting_tensor_index[i] = lhs_contracted_multi_index[i];
    }
    contracting_tensor_index[FirstContractedIndexPos] = ContractedIndexValue;
    for (size_t i = FirstContractedIndexPos + 1; i < SecondContractedIndexPos;
         i++) {
      contracting_tensor_index[i] = lhs_contracted_multi_index[i - 1];
    }
    contracting_tensor_index[SecondContractedIndexPos] = ContractedIndexValue;
    for (size_t i = SecondContractedIndexPos + 1;
         i < num_uncontracted_tensor_indices; i++) {
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
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t,
                                                    first_contracted_index::dim>
  get_storage_indices_to_sum(
      const std::index_sequence<Ints...>& /*dim_seq*/) noexcept {
    constexpr std::array<size_t, num_tensor_indices>
        lhs_contracted_multi_index =
            ContractedLhsStructure::get_canonical_tensor_index(I);
    constexpr std::array<size_t, first_contracted_index::dim>
        storage_indices_to_sum = {{UncontractedLhsStructure::get_storage_index(
            get_tensor_index_to_sum<Ints>(lhs_contracted_multi_index))...}};

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
      std::array<size_t, first_contracted_index::dim>, NumContractedComponents>
  get_sum_map(const std::index_sequence<Ints...>& /*index_seq*/) noexcept {
    constexpr std::make_index_sequence<first_contracted_index::dim> dim_seq{};
    constexpr std::array<std::array<size_t, first_contracted_index::dim>,
                         NumContractedComponents>
        map = {
            {get_storage_indices_to_sum<Ints, UncontractedLhsStructure,
                                        ContractedLhsStructure>(dim_seq)...}};
    return map;
  }

  // This is a helper that inserts the first contracted index into `LhsIndices`
  template <typename... LhsIndices>
  using get_uncontracted_lhs_tensorindex_list_helper = tmpl::append<
      tmpl::at_c<tmpl::split_at<tmpl::list<LhsIndices...>,
                                tmpl::size_t<FirstContractedIndexPos>>,
                 0>,
      tmpl::list<tmpl::at_c<ArgsList, FirstContractedIndexPos>>,
      tmpl::at_c<tmpl::split_at<tmpl::list<LhsIndices...>,
                                tmpl::size_t<FirstContractedIndexPos>>,
                 1>>;

  // This represents the uncontracted LHS indices, where you take the
  // `LhsIndices` and insert the indices that were contracted.
  //
  // e.g. If we contracted RHS R_Abac to LHS L_cb, then this means inserting
  // A and a back in to L_cb in their original spots, which is: L_Acab.
  // Specifically, `get_uncontracted_lhs_tensorindex_list` would be:
  // `tmpl::list<ti_A_t, ti_c_t, ti_a_t, ti_b_t>`
  template <typename... LhsIndices>
  using get_uncontracted_lhs_tensorindex_list = tmpl::append<
      tmpl::at_c<tmpl::split_at<get_uncontracted_lhs_tensorindex_list_helper<
                                    LhsIndices...>,
                                tmpl::size_t<SecondContractedIndexPos>>,
                 0>,
      tmpl::list<tmpl::at_c<ArgsList, SecondContractedIndexPos>>,
      tmpl::at_c<tmpl::split_at<get_uncontracted_lhs_tensorindex_list_helper<
                                    LhsIndices...>,
                                tmpl::size_t<SecondContractedIndexPos>>,
                 1>>;

  // This returns the contracted component value at the `lhs_storage_index` in
  // the contracted LHS.
  // Iterates over list of uncontracted LHS storage indices of components to
  // sum, gets the components at those indices, and returns their sum
  template <typename UncontractedLhsStructure,
            typename UncontractedLhsTensorIndexList,
            size_t NumContractedComponents, typename T1, size_t Index>
  struct ComputeContraction;

  template <typename UncontractedLhsStructure,
            typename... UncontractedLhsTensorIndices,
            size_t NumContractedComponents, typename T1, size_t Index>
  struct ComputeContraction<UncontractedLhsStructure,
                            tmpl::list<UncontractedLhsTensorIndices...>,
                            NumContractedComponents, T1, Index> {
    static SPECTRE_ALWAYS_INLINE decltype(auto) apply(
        const std::array<std::array<size_t, first_contracted_index::dim>,
                         NumContractedComponents>& map,
        const T1& t1, const size_t& lhs_storage_index) noexcept {
      return ComputeContraction<UncontractedLhsStructure,
                                tmpl::list<UncontractedLhsTensorIndices...>,
                                NumContractedComponents, decltype(t_),
                                Index + 1>::apply(map, t1, lhs_storage_index) +
             t1.template get<UncontractedLhsStructure,
                             UncontractedLhsTensorIndices...>(
                 map[lhs_storage_index][Index]);
    }
  };

  template <typename UncontractedLhsStructure,
            typename... UncontractedLhsTensorIndices,
            size_t NumContractedComponents, typename T1>
  struct ComputeContraction<
      UncontractedLhsStructure, tmpl::list<UncontractedLhsTensorIndices...>,
      NumContractedComponents, T1, first_contracted_index::dim - 1> {
    static SPECTRE_ALWAYS_INLINE decltype(auto) apply(
        const std::array<std::array<size_t, first_contracted_index::dim>,
                         NumContractedComponents>& map,
        const T1& t1, const size_t& lhs_storage_index) noexcept {
      return t1.template get<UncontractedLhsStructure,
                             UncontractedLhsTensorIndices...>(
          map[lhs_storage_index][first_contracted_index::dim - 1]);
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
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const size_t lhs_storage_index) const {
    // number of components in the contracted LHS
    constexpr size_t num_contracted_components = LhsStructure::size();
    using uncontracted_lhs_tensorindex_list =
        get_uncontracted_lhs_tensorindex_list<LhsIndices...>;

    // structure of the uncontracted LHS (LHS with contracted indices inserted)
    using uncontracted_lhs_structure =
        typename LhsTensorSymmAndIndices<ArgsList,
                                         uncontracted_lhs_tensorindex_list,
                                         Symm, IndexList>::structure;

    constexpr std::make_index_sequence<num_contracted_components> map_seq{};

    // map from contracted LHS storage indices to lists of uncontracted LHS
    // storage indices of components to sum for contraction
    constexpr std::array<std::array<size_t, first_contracted_index::dim>,
                         num_contracted_components>
        map = get_sum_map<num_contracted_components, uncontracted_lhs_structure,
                          LhsStructure>(map_seq);

    // This returns the contracted component value at the `lhs_storage_index` in
    // the contracted LHS.
    return ComputeContraction<
        uncontracted_lhs_structure, uncontracted_lhs_tensorindex_list,
        num_contracted_components, decltype(t_), 0>::apply(map, t_,
                                                           lhs_storage_index);
  }

 private:
  const std::conditional_t<std::is_base_of<Expression, T>::value, T,
                           TensorExpression<T, X, Symm, IndexList, ArgsList>>
      t_;
};

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Returns the positions of the first indices to contract in an
 * expression
 *
 * \details Given a list of values that represent an expression's generic index
 * encodings, this function looks to see if it can find a pair of values that
 * encode one generic index and the generic index with opposite valence, such as
 * `ti_A_t` and `ti_a_t`. This denotes a pair of indices that will need to be
 * contracted. If there exists more than one such pair of indices in the
 * expression, the first pair of values found will be returned.
 *
 * For example, if we have tensor \f${R^{ab}}_{ab}\f$ represented by the tensor
 * expression, `R(ti_A, ti_B, ti_a, ti_b)`, then this will return the positions
 * of the pair of values encoding `ti_A_t` and `ti_a_t`, which would be (0, 2)
 *
 * @param tensorindex_values the TensorIndex values of a tensor expression
 * @return the positions of the first pair of TensorIndex values to contract
 */
template <size_t NumIndices>
SPECTRE_ALWAYS_INLINE static constexpr std::pair<size_t, size_t>
get_first_index_positions_to_contract(
    const std::array<size_t, NumIndices>& tensorindex_values) noexcept {
  for (size_t i = 0; i < tensorindex_values.size(); ++i) {
    const size_t current_value = gsl::at(tensorindex_values, i);
    const size_t opposite_value_to_find =
        get_tensorindex_value_with_opposite_valence(current_value);
    for (size_t j = i + 1; j < tensorindex_values.size(); ++j) {
      if (opposite_value_to_find == gsl::at(tensorindex_values, j)) {
        // We found both the lower and upper version of a generic index in the
        // list of generic indices, so we return this pair's positions
        return std::pair{i, j};
      }
    }
  }
  // We couldn't find a single pair of indices that needs to be contracted
  return std::pair{std::numeric_limits<size_t>::max(),
                   std::numeric_limits<size_t>::max()};
}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Creates a contraction expression from a tensor expression if there are
 * any indices to contract
 *
 * \details If there are no indices to contract, the input TensorExpression is
 * simply returned. Otherwise, a contraction expression is created for
 * contracting one pair of upper and lower indices. If there is more than one
 * pair of indices to contract, subsequent contraction expressions are
 * recursively created, nesting one contraction expression inside another.
 *
 * For example, if we have tensor \f${R^{ab}}_{ab}\f$ represented by the tensor
 * expression, `R(ti_A, ti_B, ti_a, ti_b)`, then one contraction expression is
 * created to represent contracting \f${R^{ab}}_ab\f$ to \f${R^b}_b\f$, and a
 * second to represent contracting \f${R^b}_b\f$ to the scalar, \f${R}\f$.
 *
 * @param t the TensorExpression to potentially contract
 * @return the input tensor expression or a contraction expression of the input
 */
template <typename T, typename X, typename Symm, typename IndexList,
          typename... TensorIndices>
SPECTRE_ALWAYS_INLINE static constexpr auto contract(
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<TensorIndices...>>&
        t) noexcept {
  constexpr std::array<size_t, sizeof...(TensorIndices)> tensorindex_values = {
      {TensorIndices::value...}};
  constexpr std::pair first_index_positions_to_contract =
      get_first_index_positions_to_contract(tensorindex_values);
  constexpr std::pair no_indices_to_contract_sentinel{
      std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};

  if constexpr (first_index_positions_to_contract ==
                no_indices_to_contract_sentinel) {
    // There aren't any indices to contract, so we just return the input
    return ~t;
  } else {
    // We have a pair of indices to be contract
    return contract(
        TensorContract<first_index_positions_to_contract.first,
                       first_index_positions_to_contract.second, T, X, Symm,
                       IndexList, tmpl::list<TensorIndices...>>{t});
  }
}
}  // namespace TensorExpressions
