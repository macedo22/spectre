// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class

#pragma once

#include <array>
#include <cstddef>
#include <iostream>  // REMOVE

#include "DataStructures/Tensor/Structure.hpp"
#include "ErrorHandling/Assert.hpp"  // IWYU pragma: keep
#include "Utilities/Algorithm.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"  // IWYU pragma: keep

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Represents the indices in a TensorExpression
 *
 * \details
 * Used to denote a tensor index in a tensor slot. This allows the following
 * type of expressions to work:
 * \code{.cpp}
 * auto T = evaluate<ti_a_t, ti_b_t>(F(ti_a, ti_b) + S(ti_b, ti_a));
 * \endcode
 * where `using ti_a_t = TensorIndex<0>;` and `TensorIndex<0> ti_a;`, that is,
 * `ti_a` and `ti_b` are place holders for objects of type `TensorIndex<0>` and
 * `TensorIndex<1>` respectively.
 */
template <std::size_t I, UpLo Valence>
struct TensorIndex {
  using value_type = std::size_t;
  using type = TensorIndex<I, Valence>;
  static constexpr value_type value = I;
  static constexpr UpLo valence = Valence;
  static constexpr bool is_spacetime = I < 8;
};

/// \ingroup TensorExpressionsGroup
/// Given an `TensorIndex<size_t, UpLo>` return `TensorIndex<size_t + 1, UpLo>`
/// \tparam T the args to increment
template <typename T, UpLo Valence>
using next_tensor_index = TensorIndex<T::value + 1, Valence>;

// @{
/*!
 * \ingroup TensorExpressionsGroup
 * \brief The available TensorIndex's to use in a TensorExpression
 *
 * Available tensor indices to use in a Tensor Expression.
 * \snippet Test_TensorExpressions.cpp use_tensor_index
 */
static TensorIndex<0, UpLo::Lo> ti_a{};
static TensorIndex<0, UpLo::Up> ti_A{};
static TensorIndex<1, UpLo::Lo> ti_b{};
static TensorIndex<1, UpLo::Up> ti_B{};
static TensorIndex<2, UpLo::Lo> ti_c{};
static TensorIndex<2, UpLo::Up> ti_C{};
static TensorIndex<3, UpLo::Lo> ti_d{};
static TensorIndex<3, UpLo::Up> ti_D{};
static TensorIndex<4, UpLo::Lo> ti_e{};
static TensorIndex<4, UpLo::Up> ti_E{};
static TensorIndex<5, UpLo::Lo> ti_f{};
static TensorIndex<5, UpLo::Up> ti_F{};
static TensorIndex<6, UpLo::Lo> ti_g{};
static TensorIndex<6, UpLo::Up> ti_G{};
static TensorIndex<7, UpLo::Lo> ti_h{};
static TensorIndex<7, UpLo::Up> ti_H{};
static TensorIndex<8, UpLo::Lo> ti_i{};
static TensorIndex<8, UpLo::Up> ti_I{};
static TensorIndex<9, UpLo::Lo> ti_j{};
static TensorIndex<9, UpLo::Up> ti_J{};
static TensorIndex<10, UpLo::Lo> ti_k{};
static TensorIndex<10, UpLo::Up> ti_K{};
static TensorIndex<11, UpLo::Lo> ti_l{};
static TensorIndex<11, UpLo::Up> ti_L{};

using ti_a_t = decltype(ti_a);
using ti_A_t = decltype(ti_A);
using ti_b_t = decltype(ti_b);
using ti_B_t = decltype(ti_B);
using ti_c_t = decltype(ti_c);
using ti_C_t = decltype(ti_C);
using ti_d_t = decltype(ti_d);
using ti_D_t = decltype(ti_D);
using ti_e_t = decltype(ti_e);
using ti_E_t = decltype(ti_E);
using ti_f_t = decltype(ti_f);
using ti_F_t = decltype(ti_F);
using ti_g_t = decltype(ti_g);
using ti_G_t = decltype(ti_G);
using ti_h_t = decltype(ti_h);
using ti_H_t = decltype(ti_H);
using ti_i_t = decltype(ti_i);
using ti_I_t = decltype(ti_I);
using ti_j_t = decltype(ti_j);
using ti_J_t = decltype(ti_J);
using ti_k_t = decltype(ti_k);
using ti_K_t = decltype(ti_K);
using ti_l_t = decltype(ti_l);
using ti_L_t = decltype(ti_L);
// @}

/// \cond HIDDEN_SYMBOLS
/// \ingroup TensorExpressionsGroup
/// Type alias used when Tensor Expressions manipulate indices. These are used
/// to denote contracted as opposed to free indices.
template <int I, UpLo Valence>
using ti_contracted_t = TensorIndex<static_cast<size_t>(I + 1000), Valence>;

/// \ingroup TensorExpressionsGroup
template <int I, UpLo Valence>
TensorIndex<static_cast<size_t>(I + 1000), Valence> ti_contracted();
/// \endcond

namespace tt {
/*!
 * \ingroup TypeTraitsGroup TensorExpressionsGroup
 * \brief Check if a type `T` is a TensorIndex used in TensorExpressions
 */
template <typename T>
struct is_tensor_index : std::false_type {};
template <size_t I, UpLo Valence>
struct is_tensor_index<TensorIndex<I, Valence>> : std::true_type {};
}  // namespace tt

namespace detail {
template <typename State, typename Element, typename LHS>
struct rhs_elements_in_lhs_helper {
  using type = std::conditional_t<not std::is_same<tmpl::index_of<LHS, Element>,
                                                   tmpl::no_such_type_>::value,
                                  tmpl::push_back<State, Element>, State>;
};
}  // namespace detail

/// \ingroup TensorExpressionsGroup
/// Returns a list of all the elements in the typelist Rhs that are also in the
/// typelist Lhs.
///
/// \details
/// Given two typelists `Lhs` and `Rhs`, returns a typelist of all the elements
/// in `Rhs` that are also in `Lhs` in the same order that they are in the
/// `Rhs`.
///
/// ### Usage
/// For typelists `List1` and `List2`,
/// \code{.cpp}
/// using result = rhs_elements_in_lhs<List1, List2>;
/// \endcode
/// \metareturns
/// typelist
///
/// \semantics
/// If `Lhs = tmpl::list<A, B, C, D>` and `Rhs = tmpl::list<B, E, A>`, then
/// \code{.cpp}
/// result = tmpl::list<B, A>;
/// \endcode
template <typename Lhs, typename Rhs>
using rhs_elements_in_lhs =
    tmpl::fold<Rhs, tmpl::list<>,
               detail::rhs_elements_in_lhs_helper<tmpl::_state, tmpl::_element,
                                                  tmpl::pin<Lhs>>>;

namespace detail {
template <typename Element, typename Iteration, typename Lhs, typename Rhs,
          typename RhsWithOnlyLhs, typename IndexInLhs>
struct generate_transformation_helper {
  using tensor_index_to_find = tmpl::at<RhsWithOnlyLhs, IndexInLhs>;
  using index_to_replace_with = tmpl::index_of<Rhs, tensor_index_to_find>;
  // TODO: does it matter what valence this gets? is it just a placeholder?
  using type =
      TensorIndex<index_to_replace_with::value, tensor_index_to_find::valence>;
};

template <typename Element, typename Iteration, typename Lhs, typename Rhs,
          typename RhsWithOnlyLhs>
struct generate_transformation_helper<Element, Iteration, Lhs, Rhs,
                                      RhsWithOnlyLhs, tmpl::no_such_type_> {
  // TODO: does it matter what valence this gets? is it just a placeholder?
  using type = TensorIndex<Iteration::value, UpLo::Lo>;
};

template <typename State, typename Element, typename Iteration, typename Lhs,
          typename Rhs, typename RhsWithOnlyLhs>
struct generate_transformation_impl {
  using index_in_lhs = tmpl::index_of<Lhs, Element>;
  using type = tmpl::push_back<State, typename generate_transformation_helper<
                                          Element, Iteration, Lhs, Rhs,
                                          RhsWithOnlyLhs, index_in_lhs>::type>;
};
}  // namespace detail

/// \ingroup TensorExpressionsGroup
/// \brief Generate transformation to account for index order difference in RHS
/// and LHS.
///
/// \details
/// Generates the transformation \f$\mathcal{T}\f$ that rearranges the Tensor
/// index array to account for index order differences between the LHS and RHS
/// of the tensor expression.
///
/// ### Usage
/// For typelists `Rhs`, `Lhs` and `RhsOnyWithLhs`, where `RhsOnlyWithLhs` is
/// the result of the metafunction rhs_elements_in_lhs,
/// \code{.cpp}
/// using result = generate_transformation<Rhs, Lhs, RhsOnlyWithLhs>;
/// \endcode
/// \metareturns
/// typelist
template <typename Rhs, typename Lhs, typename RhsOnyWithLhs>
using generate_transformation = tmpl::enumerated_fold<
    Rhs, tmpl::list<>,
    detail::generate_transformation_impl<tmpl::_state, tmpl::_element, tmpl::_3,
                                         tmpl::pin<Lhs>, tmpl::pin<Rhs>,
                                         tmpl::pin<RhsOnyWithLhs>>>;

namespace detail {
template <typename Seq, typename S, typename E>
struct repeated_helper {
  using type = typename std::conditional<
      std::is_same<tmpl::count_if<Seq, std::is_same<E, tmpl::_1>>,
                   tmpl::size_t<2>>::value and
          std::is_same<tmpl::index_of<S, E>, tmpl::no_such_type_>::value,
      tmpl::push_back<S, E>, S>::type;
};

/*template <typename Seq, typename S, typename E>
struct repeated_vals_helper {
  using type = typename std::conditional<
      std::is_same<tmpl::count_if<Seq, std::is_same<E::value, tmpl::_1::value>>,
                   tmpl::size_t<2>>::value and
          std::is_same<tmpl::index_of<S, E::value>, tmpl::no_such_type_>::value,
      tmpl::push_back<S, E::value>, S>::type;
};*/

/*template <typename Seq, typename S, typename E>
struct repeated_vals_helper;
template <typename... Is, typename S, typename E>
struct repeated_vals_helper<tmpl::list<Is...>, S, E> {
  using type = typename std::conditional<
      std::is_same<tmpl::count_if<Is::value..., std::is_same<E, tmpl::_1>>,
                   tmpl::size_t<2>>::value and
          std::is_same<tmpl::index_of<S, E>, tmpl::no_such_type_>::value,
      tmpl::push_back<S, E>, S>::type;
};*/
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 * Returns a list of all the types that occurred more than once in List.
 */
template <typename List>
using repeated = tmpl::fold<
    List, tmpl::list<>,
    detail::repeated_helper<tmpl::pin<List>, tmpl::_state, tmpl::_element>>;

/*template <typename List>
using repeated_vals = tmpl::fold<
    List, tmpl::list<>,
    detail::repeated_vals_helper<
        tmpl::pin<List>, tmpl::_state, tmpl::_element>>;*/

namespace detail {
template <typename TensorIndexList, typename Element,
          typename ContractedLowerTensorIndex>
using index_replace = tmpl::replace_at<
    tmpl::replace_at<
        TensorIndexList,
        tmpl::index_of<TensorIndexList, TensorIndex<Element::value, UpLo::Lo>>,
        ContractedLowerTensorIndex>,
    tmpl::index_of<
        tmpl::replace_at<TensorIndexList,
                         tmpl::index_of<TensorIndexList,
                                        TensorIndex<Element::value, UpLo::Lo>>,
                         ContractedLowerTensorIndex>,
        TensorIndex<Element::value, UpLo::Up>>,
    TensorIndex<ContractedLowerTensorIndex::value + 1, UpLo::Up>>;

/// \cond HIDDEN_SYMBOLS
template <typename TensorIndexList, typename ReplaceTensorIndexValueList, int I>
struct replace_indices_impl
    : replace_indices_impl<
          index_replace<TensorIndexList,
                        tmpl::front<ReplaceTensorIndexValueList>,
                        ti_contracted_t<2 * I, UpLo::Lo>>,
          tmpl::pop_front<ReplaceTensorIndexValueList>, I + 1> {};
/// \endcond

template <typename TensorIndexList, int I>
struct replace_indices_impl<TensorIndexList, tmpl::list<>, I> {
  using type = TensorIndexList;
};
}  // namespace detail

namespace TensorExpressions {
/*!
 * \ingroup TensorExpressionsGroup
 * \brief Determines and stores a LHS tensor's symmetry and index list from a
 * RHS tensor expression and desired LHS index order
 *
 * \details Given the generic index order of a RHS TensorExpression and the
 * generic index order of the desired LHS Tensor, this creates a mapping between
 * the two that is used to determine the (potentially reordered) ordering of the
 * elements of the desired LHS Tensor`s ::Symmetry and typelist of
 * \ref SpacetimeIndex "TensorIndexType"s.
 *
 * @tparam RhsTensorIndexList the typelist of TensorIndex of the RHS
 * TensorExpression, e.g. `ti_a_t`, `ti_b_t`, `ti_c_t`
 * @tparam LhsTensorIndexList the typelist of TensorIndexs of the desired LHS
 * tensor, e.g. `ti_b_t`, `ti_c_t`, `ti_a_t`
 * @tparam RhsSymmetry the ::Symmetry of the RHS indices
 * @tparam RhsTensorIndexTypeList the RHS TensorExpression's typelist of
 * \ref SpacetimeIndex "TensorIndexType"s
 */
template <typename RhsTensorIndexList, typename LhsTensorIndexList,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          size_t RhsNumIndices = tmpl::size<RhsTensorIndexList>::value,
          size_t LhsNumIndices = tmpl::size<LhsTensorIndexList>::value,
          typename RhsIndexSequence = std::make_index_sequence<RhsNumIndices>,
          typename LhsIndexSequence = std::make_index_sequence<LhsNumIndices>>
struct LhsTensor;

template <typename RhsTensorIndexList, typename... LhsTensorIndices,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          size_t RhsNumIndices, size_t LhsNumIndices, size_t... RhsInts,
          size_t... LhsInts>
struct LhsTensor<RhsTensorIndexList, tmpl::list<LhsTensorIndices...>,
                 RhsSymmetry, RhsTensorIndexTypeList, RhsNumIndices,
                 LhsNumIndices, std::index_sequence<RhsInts...>,
                 std::index_sequence<LhsInts...>> {
  static constexpr std::array<size_t, LhsNumIndices> lhs_tensorindex_values = {
      {LhsTensorIndices::value...}};
  static constexpr std::array<size_t, RhsNumIndices> rhs_tensorindex_values = {
      {tmpl::at_c<RhsTensorIndexList, RhsInts>::value...}};
  static constexpr std::array<size_t, LhsNumIndices> lhs_to_rhs_map = {
      {std::distance(rhs_tensorindex_values.begin(),
                     alg::find(rhs_tensorindex_values,
                               lhs_tensorindex_values[LhsInts]))...}};

  // Desired LHS Tensor's Symmetry and typelist of TensorIndexTypes
  using symmetry =
      Symmetry<tmpl::at_c<RhsSymmetry, lhs_to_rhs_map[LhsInts]>::value...>;
  using tensorindextype_list = tmpl::list<
      tmpl::at_c<RhsTensorIndexTypeList, lhs_to_rhs_map[LhsInts]>...>;
  using structure = Tensor_detail::Structure<
      symmetry, tmpl::at_c<RhsTensorIndexTypeList, lhs_to_rhs_map[LhsInts]>...>;
};
}  // namespace TensorExpressions

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename TensorIndexList, typename ReplaceTensorIndexValueList>
using replace_indices =
    typename detail::replace_indices_impl<TensorIndexList,
                                          ReplaceTensorIndexValueList, 0>::type;

/// \ingroup TensorExpressionsGroup
/// \brief Marks a class as being a TensorExpression
///
/// \details
/// The empty base class that all TensorExpression`s must inherit from.
/// \derivedrequires
/// 1) The args_list will be the sorted args_list received as input
///
/// 2) The tensor indices will be swapped to conform with mathematical notation
struct Expression {};

/// \cond
template <typename DataType, typename Symm, typename IndexList>
class Tensor;
/// \endcond

// @{
/// \ingroup TensorExpressionsGroup
/// \brief The base class all tensor expression implementations derive from
///
/// \tparam Derived the derived class needed for
/// [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
/// \tparam DataType the type of the data being stored in the Tensor's
/// \tparam Symm the ::Symmetry of the Derived class
/// \tparam IndexList the list of \ref SpacetimeIndex "TensorIndex"'s
/// \tparam Args the tensor indices, e.g. `_a` and `_b` in `F(_a, _b)`
/// \cond HIDDEN_SYMBOLS
template <typename Derived, typename DataType, typename Symm,
          typename IndexList, typename Args = tmpl::list<>,
          typename ReducedArgs = tmpl::list<>>
struct TensorExpression;
/// \endcond

template <typename Derived, typename DataType, typename Symm,
          typename... Indices, template <typename...> class ArgsList,
          typename... Args>
struct TensorExpression<Derived, DataType, Symm, tmpl::list<Indices...>,
                        ArgsList<Args...>> : public Expression {
  static_assert(sizeof...(Args) == 0 or
                    sizeof...(Args) ==
                        tmpl::size<tmpl::list<Indices...>>::value,
                "the number of Tensor indices must match the number of "
                "components specified in an expression.");
  using type = DataType;
  using symmetry = Symm;
  using index_list = tmpl::list<Indices...>;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  /// Typelist of the tensor indices, e.g. `_a_t` and `_b_t` in `F(_a, _b)`
  using args_list = ArgsList<Args...>;
  using structure = Tensor_detail::Structure<symmetry, Indices...>;

  // @{
  /// If Derived is a TensorExpression, it is casted down to the derived
  /// class. This is enabled by the
  /// [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
  ///
  /// Otherwise, it is a Tensor. Since Tensor is not derived from
  /// TensorExpression (because of complications arising from the indices being
  /// part of the expression, specifically Tensor may need to derive off of
  /// hundreds or thousands of base classes, which is not feasible), return a
  /// reference to a TensorExpression, which has a sufficient interface to
  /// evaluate the expression.
  ///
  /// \returns const TensorExpression<Derived, DataType, Symm, IndexList,
  /// ArgsList<Args...>>&
  SPECTRE_ALWAYS_INLINE const auto& operator~() const noexcept {
    if constexpr (tt::is_a_v<Tensor, Derived>) {
      return *this;
    } else {
      return static_cast<const Derived&>(*this);
    }
  }

  // @}

  // @{
  /// \cond HIDDEN_SYMBOLS
  /// \ingroup TensorExpressionsGroup
  /// Helper struct to compute the correct tensor index array from a
  /// typelist of std::integral_constant's indicating the ordering. This is
  /// needed for dealing with expressions such as \f$T_{ab} = F_{ba}\f$ and gets
  /// the ordering on the RHS to be correct compared with where the indices are
  /// on the LHS.
  template <typename U>
  struct ComputeCorrectTensorIndex;

  template <template <typename...> class RedArgsList, typename... RedArgs>
  struct ComputeCorrectTensorIndex<RedArgsList<RedArgs...>> {
    template <typename U, std::size_t Size>
    SPECTRE_ALWAYS_INLINE static constexpr std::array<U, Size> apply(
        const std::array<U, Size>& tensor_index) {
      return std::array<U, Size>{{tensor_index[RedArgs::value]...}};
    }
  };
  /// \endcond
  // @}

  /// \brief return the value of type DataType with tensor index `tensor_index`
  ///
  /// \details
  /// If Derived is a TensorExpression, `tensor_index` is forwarded onto the
  /// next TensorExpression.
  ///
  /// Otherwise, it is a Tensor, where one big challenge with TensorExpression
  /// implementation is the reordering of the Indices on the RHS and LHS of the
  /// expression. This algorithm implemented in ::rhs_elements_in_lhs and
  /// ::generate_transformation handles the index sorting.
  ///
  /// Here are some examples of what the algorithm does:
  ///
  /// LhsIndices is the desired ordering.
  ///
  /// LHS:
  /// \code
  /// <0, 1>
  /// \endcode
  /// RHS:
  /// \code
  /// <1, 2, 3, 0> -Transform> <3, 1, 2, 0>
  /// \endcode
  ///
  /// LHS:
  /// \code
  /// <0, 1, 2> <a, b, c>
  /// \endcode
  /// RHS:
  /// \code
  /// <2, 0, 1> -Transform> <2 , 1, 0>
  /// \endcode
  ///
  /// Below is pseudo-code of the algorithm written in a non-functional way
  /// \verbatim
  /// for Element in RHS:
  ///   if (Element in LHS):
  ///     index_in_LHS = index_of<LHS, Element>
  ///     tensor_index_to_find = at<RHS_with_only_LHS, index_in_LHS>
  ///     index_to_replace_with = index_of<RHS, tensor_index_to_find>
  ///     T_RHS = push_back<T_RHS, index_to_replace_with>
  ///   else:
  ///     T_RHS = push_back<T_RHS, iteration>
  ///   endif
  /// end for
  /// \endverbatim
  ///
  /// \tparam LhsIndices the tensor indices on the LHS on the expression
  /// \param tensor_index the tensor component to retrieve
  /// \return the value of the DataType of component `tensor_index`
  template <typename... LhsIndices, typename ArrayValueType>
  SPECTRE_ALWAYS_INLINE type
  get(const std::array<ArrayValueType, num_tensor_indices>& tensor_index)
      const noexcept {
    if constexpr (tt::is_a_v<Tensor, Derived>) {
      ASSERT(t_ != nullptr,
             "A TensorExpression that should be holding a pointer to a Tensor "
             "is holding a nullptr.");
      using rhs = args_list;
      // To deal with Tensor products we need the ordering of only the subset of
      // tensor indices present in this term
      using lhs = rhs_elements_in_lhs<rhs, tmpl::list<LhsIndices...>>;
      using rhs_only_with_lhs = rhs_elements_in_lhs<lhs, rhs>;
      using transformation =
          generate_transformation<rhs, lhs, rhs_only_with_lhs>;
      return t_->get(
          ComputeCorrectTensorIndex<transformation>::apply(tensor_index));
    } else {
      ASSERT(t_ == nullptr,
             "A TensorExpression that shouldn't be holding a pointer to a "
             "Tensor is holding one.");
      return (~*this).template get<LhsIndices...>(tensor_index);
    }
  }

  /// \brief Computes the right hand side tensor multi-index that corresponds to
  /// the left hand side tensor multi-index, according to their generic indices
  ///
  /// \details
  /// Given the order of the generic indices for the left hand side (LHS) and
  /// right hand side (RHS) and a specific LHS tensor multi-index, the
  /// computation of the equivalent multi-index for the RHS tensor accounts for
  /// differences in the ordering of the generic indices on the LHS and RHS.
  ///
  /// Here, the elements of `lhs_index_order` and `rhs_index_order` refer to
  /// TensorIndex::values that correspond to generic tensor indices,
  /// `lhs_tensor_multi_index` is a multi-index for the LHS tensor, and the
  /// equivalent RHS tensor multi-index is returned. If we have LHS tensor
  /// \f$L_{ab}\f$, RHS tensor \f$R_{ba}\f$, and the LHS component \f$L_{31}\f$,
  /// the corresponding RHS component is \f$R_{13}\f$.
  ///
  /// Here is an example of what the algorithm does:
  ///
  /// `lhs_index_order`:
  /// \code
  /// [0, 1, 2] // i.e. abc
  /// \endcode
  /// `rhs_index_order`:
  /// \code
  /// [1, 2, 0] // i.e. bca
  /// \endcode
  /// `lhs_tensor_multi_index`:
  /// \code
  /// [4, 0, 3] // i.e. a = 4, b = 0, c = 3
  /// \endcode
  /// returned RHS tensor multi-index:
  /// \code
  /// [0, 3, 4] // i.e. b = 0, c = 3, a = 4
  /// \endcode
  ///
  /// \tparam NumberOfIndices the number of indices in the RHS tensor
  /// \param lhs_index_order the generic index order of the LHS tensor
  /// \param rhs_index_order the generic index order of the RHS tensor
  /// \param lhs_tensor_multi_index the specific LHS tensor multi-index
  /// \return the RHS tensor multi-index that corresponds to
  /// `lhs_tensor_multi_index`, according to the index orders in
  /// `lhs_index_order` and `rhs_index_order`
  template <size_t NumberOfIndices>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, NumberOfIndices>
  compute_rhs_tensor_index(
      const std::array<size_t, NumberOfIndices>& lhs_index_order,
      const std::array<size_t, NumberOfIndices>& rhs_index_order,
      const std::array<size_t, NumberOfIndices>&
          lhs_tensor_multi_index) noexcept {
    std::array<size_t, NumberOfIndices> rhs_tensor_multi_index{};
    for (size_t i = 0; i < NumberOfIndices; ++i) {
      rhs_tensor_multi_index[static_cast<unsigned long>(
          std::distance(rhs_index_order.begin(),
                        alg::find(rhs_index_order, lhs_index_order[i])))] =
          lhs_tensor_multi_index[i];
    }
    return rhs_tensor_multi_index;
  }

  /// \brief Computes a mapping from the storage indices of the left hand side
  /// tensor to the right hand side tensor
  /// \tparam LhsStructure the Structure of the Tensor on the left hand side of
  /// the TensorExpression
  /// \tparam LhsIndices the TensorIndexs of the Tensor on the left hand side
  /// \return the mapping from the left hand side to the right hand side storage
  /// indices
  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t,
                                                    LhsStructure::size()>
  compute_lhs_to_rhs_map() noexcept {
    constexpr size_t num_components = LhsStructure::size();
    constexpr size_t rank = LhsStructure::rank();
    std::array<size_t, num_components> lhs_to_rhs_map{};
    auto LhsStorageToTensorIndices = LhsStructure::storage_to_tensor_index();
    for (size_t lhs_storage_index = 0; lhs_storage_index < num_components;
         ++lhs_storage_index) {
      // `compute_rhs_tensor_index` will return the RHS tensor multi-index that
      // corresponds to the LHS tensor multi-index, according to the order of
      // the generic indices for the LHS and RHS. structure::get_storage_index
      // will then get the RHS storage index that corresponds to this RHS
      // tensor multi-index.
      lhs_to_rhs_map[lhs_storage_index] =
          structure::get_storage_index(compute_rhs_tensor_index<rank>(
              {{LhsIndices::value...}}, {{Args::value...}},
              LhsStorageToTensorIndices[lhs_storage_index]));
    }
    return lhs_to_rhs_map;
  }

  /// \brief return the value at a left hand side tensor's storage index
  ///
  /// \details
  /// If Derived is a TensorExpression, `storage_index` is forwarded onto the
  /// next TensorExpression.
  ///
  /// Otherwise, it is a Tensor, where one big challenge with TensorExpression
  /// implementation is the reordering of the indices on the left hand side
  /// (LHS) and right hand side (RHS) of the expression. The algoritms
  /// implemented in `compute_lhs_to_rhs_map` and `compute_rhs_tensor_index`
  /// handle the index sorting by mapping between the generic index orders of
  /// the LHS and RHS.
  ///
  /// \tparam LhsStructure the Structure of the Tensor on the LHS of the
  /// TensorExpression
  /// \tparam LhsIndices the TensorIndexs of the Tensor on the LHS of the tensor
  /// expression, e.g. `ti_a_t`, `ti_b_t`, `ti_c_t`
  /// \param lhs_storage_index the storage index of the LHS tensor component to
  /// retrieve
  /// \return the value of the DataType of the component at `lhs_storage_index`
  /// in the LHS tensor
  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE type
  get(const size_t lhs_storage_index) const noexcept {
    if constexpr (not tt::is_a_v<Tensor, Derived>) {
      return static_cast<const Derived&>(*this)
          .template get<LhsStructure, LhsIndices...>(lhs_storage_index);
    } else if constexpr (sizeof...(LhsIndices) < 2) {
      return (*t_)[lhs_storage_index];
    } else {
      constexpr std::array<size_t, LhsStructure::size()> lhs_to_rhs_map =
          compute_lhs_to_rhs_map<LhsStructure, LhsIndices...>();
      return (*t_)[lhs_to_rhs_map[lhs_storage_index]];
    }
  }

  /// Retrieve the i'th entry of the Tensor being held
  template <typename V = Derived,
            Requires<tt::is_a<Tensor, V>::value> = nullptr>
  SPECTRE_ALWAYS_INLINE type operator[](const size_t i) const {
    return t_->operator[](i);
  }

  /// \brief Construct a TensorExpression from another TensorExpression.
  ///
  /// In this case we do not need to store a pointer to the TensorExpression
  /// since we can cast back to the derived class using operator~.
  template <typename V = Derived,
            Requires<not tt::is_a<Tensor, V>::value> = nullptr>
  TensorExpression() {}  // NOLINT

  /// \brief Construct a TensorExpression from a Tensor.
  ///
  /// We need to store a pointer to the Tensor in a member variable in order
  /// to be able to access the data when later evaluating the tensor expression.
  explicit TensorExpression(const Tensor<DataType, Symm, index_list>& t)
      : t_(&t) {}

 private:
  /// Holds a pointer to a Tensor if the TensorExpression represents one.
  ///
  /// The pointer is needed so that the Tensor class need not derive from
  /// TensorExpression. The reason deriving off of TensorExpression is
  /// problematic for Tensor is that the index structure is part of the type
  /// of the TensorExpression, so every possible permutation and combination of
  /// indices must be derived from. For a rank-3 tensor this is already over 500
  /// base classes, which the Intel compiler takes too long to compile.
  ///
  /// Benchmarking shows that GCC 6 and Clang 3.9.0 can derive off of 672 base
  /// classes with compilation time of about 5 seconds, while the Intel compiler
  /// v16.3 takes around 8 minutes. These tests were done on a Haswell Core i5.
  const Derived* t_ = nullptr;
};
// @}
