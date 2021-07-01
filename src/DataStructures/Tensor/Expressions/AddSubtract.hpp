// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for adding and subtracting tensors

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/SpatialSpacetimeIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndexTransformation.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace TensorExpressions {
template <typename T1, typename T2, typename ArgsList1, typename ArgsList2,
          int Sign>
struct AddSub;
}  // namespace TensorExpressions
template <typename Derived, typename DataType, typename Symm,
          typename IndexList, typename Args, typename ReducedArgs>
struct TensorExpression;
/// \endcond

namespace TensorExpressions {

namespace detail {
// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the canonical symmetry of the tensor resulting from
/// adding or subtracting two tensors, according to their symmetries
///
/// \details The canonical symmetry returned follows the convention defined by
/// ::Symmetry: symmetry values are in ascending order from right to left. If
/// the convention implemented by ::Symmetry changes, this function will also
/// need to be updated to match the new convention.
///
/// This function treats the two input symmetries as aligned (i.e. each position
/// of `symm1` and `symm2` corresponds to a shared generic index at that
/// position). The resultant symmetry is determined as follows: indices that are
/// symmetric in both input symmetries are also symmetric in the resultant
/// tensor.
///
/// \param symm1 the symmetry of the first tensor being added or subtracted
/// \param symm2 the symmetry of the second tensor being added or subtracted
/// \return the canonical symmetry of the tensor resulting from adding or
/// subtracting two tensors
template <size_t NumIndices, Requires<(NumIndices >= 2)> = nullptr>
constexpr std::array<std::int32_t, NumIndices> get_addsub_symm(
    const std::array<std::int32_t, NumIndices>& symm1,
    const std::array<std::int32_t, NumIndices>& symm2) {
  std::array<std::int32_t, NumIndices> addsub_symm{};
  size_t right_index = NumIndices - 1;
  std::int32_t symm_value_to_set = 1;

  while (right_index < NumIndices) {
    std::int32_t symm1_value_to_find = symm1[right_index];
    std::int32_t symm2_value_to_find = symm2[right_index];
    // if we haven't yet set right_index for the resultant symmetry
    if (addsub_symm[right_index] == 0) {
      addsub_symm[right_index] = symm_value_to_set;
      for (size_t left_index = right_index - 1; left_index < NumIndices;
           left_index--) {
        // if left_index of the resultant symmetry is not yet set and we've
        // found a common symmetry between symm1 and symm2 at this index
        if (addsub_symm[left_index] == 0 and
            symm1[left_index] == symm1_value_to_find and
            symm2[left_index] == symm2_value_to_find) {
          addsub_symm[left_index] = symm_value_to_set;
        }
      }
      symm_value_to_set++;
    }
    right_index--;
  }

  return addsub_symm;
}

template <size_t NumIndices, Requires<(NumIndices < 2)> = nullptr>
constexpr std::array<std::int32_t, NumIndices> get_addsub_symm(
    const std::array<std::int32_t, NumIndices>& symm1,
    const std::array<std::int32_t, NumIndices>& /*symm2*/) {
  return symm1;
}
// @}

/// \ingroup TensorExpressionsGroup
/// \brief Helper struct for computing the canonical symmetry of the tensor
/// resulting from adding or subtracting two tensors, according to their
/// symmetries and generic index orders
///
/// \details The resultant symmetry (`type`) values correspond to the index
/// order of the first tensor operand being added or subtracted:
/// `TensorIndexList1`.
///
/// \tparam SymmList1 the ::Symmetry of the first operand
/// \tparam SymmList2 the ::Symmetry of the second operand
/// \tparam TensorIndexList1 the generic indices of the first operand
/// \tparam TensorIndexList2 the generic indices of the second operand
template <typename SymmList1, typename SymmList2, typename TensorIndexList1,
          typename TensorIndexList2,
          size_t NumIndices = tmpl::size<SymmList1>::value,
          typename IndexSequence = std::make_index_sequence<NumIndices>>
struct AddSubSymmetry;

template <template <typename...> class SymmList1, typename... Symm1,
          template <typename...> class SymmList2, typename... Symm2,
          template <typename...> class TensorIndexList1,
          typename... TensorIndices1,
          template <typename...> class TensorIndexList2,
          typename... TensorIndices2, size_t NumIndices, size_t... Ints>
struct AddSubSymmetry<SymmList1<Symm1...>, SymmList2<Symm2...>,
                      TensorIndexList1<TensorIndices1...>,
                      TensorIndexList2<TensorIndices2...>, NumIndices,
                      std::index_sequence<Ints...>> {
  static constexpr std::array<size_t, NumIndices> lhs_tensorindex_values = {
      {TensorIndices1::value...}};
  static constexpr std::array<size_t, NumIndices> rhs_tensorindex_values = {
      {TensorIndices2::value...}};
  static constexpr std::array<size_t, NumIndices> lhs_to_rhs_map = {
      {std::distance(
          rhs_tensorindex_values.begin(),
          alg::find(rhs_tensorindex_values, lhs_tensorindex_values[Ints]))...}};

  static constexpr std::array<std::int32_t, NumIndices> symm1 = {
      {Symm1::value...}};
  static constexpr std::array<std::int32_t, NumIndices> symm2 = {
      {Symm2::value...}};
  // 2nd argument is symm2 rearranged according to `TensorIndexList1` order
  // so that the two symmetry arguments to `get_addsub_symm` are aligned
  // w.r.t. their generic index orders
  static constexpr std::array<std::int32_t, NumIndices> addsub_symm =
      get_addsub_symm(symm1, {{symm2[lhs_to_rhs_map[Ints]]...}});

  using type = tmpl::integral_list<std::int32_t, addsub_symm[Ints]...>;
};

/// \ingroup TensorExpressionsGroup
/// \brief Helper struct for defining the symmetry, index list, and
/// generic index list of the tensor resulting from adding or
/// subtracting two tensor expressions
///
/// \tparam T1 the first tensor expression operand
/// \tparam T2 the second tensor expression operand
template <typename T1, typename T2>
struct AddSubType {
  static_assert(std::is_base_of_v<Expression, T1> and
                    std::is_base_of_v<Expression, T2>,
                "Parameters to AddSubType must be TensorExpressions");
  using type =
      std::conditional_t<std::is_same<typename T1::type, DataVector>::value or
                             std::is_same<typename T2::type, DataVector>::value,
                         DataVector, double>;
  using symmetry =
      typename AddSubSymmetry<typename T1::symmetry, typename T2::symmetry,
                              typename T1::args_list,
                              typename T2::args_list>::type;
  using index_list = typename T1::index_list;
  using tensorindex_list = typename T1::args_list;
};

/// \brief Helper struct for checking that an index in one operand can be added
/// to and subtracted from its corresponding index in another operand
///
/// \details
/// Corresponding indices between two operands are marked by using the same
/// generic index, such as `ti_a`. For it to be possible to add or subtract one
/// operand's index to its corresponding index in another operand, this checks
/// that the following is true for the index in both operands:
/// - has the same valence (`UpLo`)
/// - has the same `Frame` type
/// - has the same number of spatial dimensions (allowing for expressions that
///   use generic spatial indices for spacetime indices on either side)
///
/// \tparam IndexList1 the first operand's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam IndexList2 the second operand's
/// \ref SpacetimeIndex "TensorIndexType" list
/// \tparam TensorIndexList1 the first operand's generic index list
/// \tparam TensorIndexList2 the second operand's generic index list
/// \tparam CurrentTensorIndex1 the first operand's generic index that is being
/// checked, e.g. the type of `ti_a`
template <typename IndexList1, typename IndexList2, typename TensorIndexList1,
          typename TensorIndexList2, typename CurrentTensorIndex1>
struct AddSubIndexCheckHelper {
  using index1 =
      tmpl::at<IndexList1,
               tmpl::index_of<TensorIndexList1, CurrentTensorIndex1>>;
  using index2 =
      tmpl::at<IndexList2,
               tmpl::index_of<TensorIndexList2, CurrentTensorIndex1>>;

  using type = std::integral_constant<
      bool,
      index1::ul == index2::ul and
          std::is_same_v<typename index1::Frame, typename index2::Frame> and
          ((index1::index_type == index2::index_type and
            index1::dim == index2::dim) or
           (index1::index_type == IndexType::Spacetime and
            index1::dim == index2::dim + 1) or
           (index2::index_type == IndexType::Spacetime and
            index1::dim + 1 == index2::dim))>;
};

/// \brief Check that the indices of the two operands of an `AddSub` expression
/// can be added and subtracted
///
/// \details
/// For more details, see `AddSubIndexCheckHelper`, which performs the check for
/// each index one at a time.
///
/// \tparam IndexList1 the first operand's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam IndexList2 the second operand's
/// \ref SpacetimeIndex "TensorIndexType" list
/// \tparam TensorIndexList1 the first operand's generic index list
/// \tparam TensorIndexList2 the second operand's generic index list
template <typename IndexList1, typename IndexList2, typename TensorIndexList1,
          typename TensorIndexList2>
using AddSubIndexCheck = tmpl::fold<
    TensorIndexList1, tmpl::bool_<true>,
    tmpl::and_<tmpl::_state, AddSubIndexCheckHelper<
                                 tmpl::pin<IndexList1>, tmpl::pin<IndexList2>,
                                 tmpl::pin<TensorIndexList1>,
                                 tmpl::pin<TensorIndexList2>, tmpl::_element>>>;
}  // namespace detail

template <typename T1, typename T2, typename ArgsList1, typename ArgsList2,
          int Sign>
struct AddSub;

template <typename T1, typename T2, template <typename...> class ArgsList1,
          template <typename...> class ArgsList2, typename... Args1,
          typename... Args2, int Sign>
struct AddSub<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>, Sign>
    : public TensorExpression<
          AddSub<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>, Sign>,
          typename detail::AddSubType<T1, T2>::type,
          typename detail::AddSubType<T1, T2>::symmetry,
          typename detail::AddSubType<T1, T2>::index_list,
          typename detail::AddSubType<T1, T2>::tensorindex_list> {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value or
                    std::is_same<T1, NumberAsExpression>::value or
                    std::is_same<T2, NumberAsExpression>::value,
                "Cannot add or subtract Tensors holding different data types.");
  static_assert(
      detail::AddSubIndexCheck<typename T1::index_list, typename T2::index_list,
                               ArgsList1<Args1...>, ArgsList2<Args2...>>::value,
      "You are attempting to add indices of different types, e.g. T^a_b + "
      "S^b_a, which doesn't make sense. The indices may also be in different "
      "frames, different types (spatial vs. spacetime) or of different "
      "dimension.");
  static_assert(Sign == 1 or Sign == -1,
                "Invalid Sign provided for addition or subtraction of Tensor "
                "elements. Sign must be 1 (addition) or -1 (subtraction).");

  using type = typename detail::AddSubType<T1, T2>::type;
  using symmetry = typename detail::AddSubType<T1, T2>::symmetry;
  using index_list = typename detail::AddSubType<T1, T2>::index_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  using args_list = typename T1::args_list;
  static constexpr std::array<size_t, num_tensor_indices>
      operand_index_transformation =
          compute_tensorindex_transformation<num_tensor_indices>(
              {{Args1::value...}}, {{Args2::value...}});
  // positions of indices in first operand where generic spatial indices are
  // used for spacetime indices
  static constexpr auto first_op_spatial_spacetime_index_positions =
      detail::get_spatial_spacetime_index_positions<typename T1::index_list,
                                                    ArgsList1<Args1...>>();
  // positions of indices in second operand where generic spatial indices are
  // used for spacetime indices
  static constexpr auto second_op_spatial_spacetime_index_positions =
      detail::get_spatial_spacetime_index_positions<typename T2::index_list,
                                                    ArgsList2<Args2...>>();

  AddSub(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}
  ~AddSub() override = default;

  template <typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& lhs_multi_index) const {
    if constexpr (std::is_same_v<tmpl::list<Args1...>, tmpl::list<Args2...>>) {
      if constexpr (first_op_spatial_spacetime_index_positions.size() == 0) {
        // either:
        // (i) neither operand uses a generic spatial index for a spacetime
        // index, or
        // (ii) only the second operand uses a generic spatial index for a
        // spacetime index
        std::array<size_t, num_tensor_indices> second_op_multi_index =
            lhs_multi_index;
        for (size_t i = 0;
             i < second_op_spatial_spacetime_index_positions.size(); i++) {
          gsl::at(second_op_multi_index, i) =
              gsl::at(lhs_multi_index,
                      gsl::at(second_op_spatial_spacetime_index_positions, i)) +
              1;
        }
        if constexpr (Sign == 1) {
          return t1_.template get<Args1...>(lhs_multi_index) +
                 t2_.template get<Args2...>(second_op_multi_index);
        } else {
          return t1_.template get<Args1...>(lhs_multi_index) -
                 t2_.template get<Args2...>(second_op_multi_index);
        }
      } else {
        // either:
        // (i) only the first operand uses a generic spatial index for a
        // spacetime index
        // (ii) both operands use a generic spatial index for a spacetime index
        std::array<size_t, num_tensor_indices> second_op_multi_index{};
        for (size_t i = 0;
             i < first_op_spatial_spacetime_index_positions.size(); i++) {
          gsl::at(second_op_multi_index, i) =
              gsl::at(lhs_multi_index,
                      gsl::at(first_op_spatial_spacetime_index_positions, i)) -
              1;
        }
        for (size_t i = 0;
             i < second_op_spatial_spacetime_index_positions.size(); i++) {
          gsl::at(second_op_multi_index, i) =
              gsl::at(lhs_multi_index,
                      gsl::at(second_op_spatial_spacetime_index_positions, i)) +
              1;
        }
        if constexpr (Sign == 1) {
          return t1_.template get<Args1...>(lhs_multi_index) +
                 t2_.template get<Args2...>(second_op_multi_index);
        } else {
          return t1_.template get<Args1...>(lhs_multi_index) -
                 t2_.template get<Args2...>(second_op_multi_index);
        }
      }
    } else {
      if constexpr (first_op_spatial_spacetime_index_positions.size() == 0) {
        // either:
        // (i) neither operand uses a generic spatial index for a spacetime
        // index, or
        // (ii) only the second operand uses a generic spatial index for a
        // spacetime index
        auto second_op_multi_index = transform_multi_index(
            lhs_multi_index, operand_index_transformation);
        for (size_t i = 0;
             i < second_op_spatial_spacetime_index_positions.size(); i++) {
          gsl::at(second_op_multi_index,
                  gsl::at(second_op_spatial_spacetime_index_positions, i)) += 1;
        }
        if constexpr (Sign == 1) {
          return t1_.template get<Args1...>(lhs_multi_index) +
                 t2_.template get<Args2...>(second_op_multi_index);
        } else {
          return t1_.template get<Args1...>(lhs_multi_index) -
                 t2_.template get<Args2...>(second_op_multi_index);
        }
      } else {
        // either:
        // (i) only the first operand uses a generic spatial index for a
        // spacetime index
        // (ii) both operands use a generic spatial index for a spacetime index
        std::array<size_t, num_tensor_indices> second_op_multi_index =
            lhs_multi_index;
        for (size_t i = 0;
             i < first_op_spatial_spacetime_index_positions.size(); i++) {
          gsl::at(second_op_multi_index,
                  gsl::at(first_op_spatial_spacetime_index_positions, i)) -= 1;
          // gsl::at(lhs_multi_index,
          //         gsl::at(lhs_spatial_spacetime_index_positions, j)) -= 1;
        }
        for (size_t i = 0;
             i < second_op_spatial_spacetime_index_positions.size(); i++) {
          gsl::at(second_op_multi_index,
                  gsl::at(operand_index_transformation,
                          gsl::at(second_op_spatial_spacetime_index_positions,
                                  i))) += 1;
          // gsl::at(rhs_multi_index,
          //         gsl::at(rhs_spatial_spacetime_index_positions, j)) += 1;
        }
        if constexpr (Sign == 1) {
          return t1_.template get<Args1...>(lhs_multi_index) +
                 t2_.template get<Args2...>(transform_multi_index(
                     second_op_multi_index, operand_index_transformation));
        } else {
          return t1_.template get<Args1...>(lhs_multi_index) -
                 t2_.template get<Args2...>(transform_multi_index(
                     second_op_multi_index, operand_index_transformation));
        }
      }
    }
  }

  SPECTRE_ALWAYS_INLINE typename T1::type operator[](size_t i) const {
    if constexpr (Sign == 1) {
      return t1_[i] + t2_[i];
    } else {
      return t1_[i] - t2_[i];
    }
  }

 private:
  T1 t1_;
  T2 t2_;
};
}  // namespace TensorExpressions

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename T1, typename T2, typename X1, typename X2, typename Symm1,
          typename Symm2, typename IndexList1, typename IndexList2,
          typename Args1, typename Args2>
SPECTRE_ALWAYS_INLINE auto operator+(
    const TensorExpression<T1, X1, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X2, Symm2, IndexList2, Args2>& t2) {
  static_assert(tmpl::size<Args1>::value == tmpl::size<Args2>::value,
                "Tensor addition is only possible with the same rank tensors");
  static_assert(tmpl::equal_members<Args1, Args2>::value,
                "The indices when adding two tensors must be equal. This error "
                "occurs from expressions like A(_a, _b) + B(_c, _a)");
  return TensorExpressions::AddSub<T1, T2, Args1, Args2, 1>(~t1, ~t2);
}

// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the sum of a tensor
/// expression and a `double`
///
/// \details
/// The tensor expression operand must represent an expression that, when
/// evaluated, would be a rank 0 tensor. For example, if `R` and `S` are
/// Tensors, here is a non-exhaustive list of some of the acceptable forms that
/// the tensor expression operand could take:
/// - `R()`
/// - `R(ti_A, ti_a)`
/// - `(R(ti_A, ti_B) * S(ti_a, ti_b))`
///
/// \tparam T the derived TensorExpression type of the tensor expression operand
/// of the sum
/// \tparam X the type of data stored in the tensor expression operand of the
/// sum
/// \param t the tensor expression operand of the sum
/// \param number the `double` operand of the sum
/// \return the tensor expression representing the sum of a tensor expression
/// and a `double`
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator+(
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>, tmpl::list<>>& t,
    const double number) {
  return t + TensorExpressions::NumberAsExpression(number);
}
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator+(
    const double number,
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>, tmpl::list<>>& t) {
  return TensorExpressions::NumberAsExpression(number) + t;
}
// @}

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename T1, typename T2, typename X1, typename X2, typename Symm1,
          typename Symm2, typename IndexList1, typename IndexList2,
          typename Args1, typename Args2>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T1, X1, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X2, Symm2, IndexList2, Args2>& t2) {
  static_assert(tmpl::size<Args1>::value == tmpl::size<Args2>::value,
                "Tensor addition is only possible with the same rank tensors");
  static_assert(tmpl::equal_members<Args1, Args2>::value,
                "The indices when adding two tensors must be equal. This error "
                "occurs from expressions like A(_a, _b) - B(_c, _a)");
  return TensorExpressions::AddSub<T1, T2, Args1, Args2, -1>(~t1, ~t2);
}

// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the difference of a tensor
/// expression and a `double`
///
/// \details
/// The tensor expression operand must represent an expression that, when
/// evaluated, would be a rank 0 tensor. For example, if `R` and `S` are
/// Tensors, here is a non-exhaustive list of some of the acceptable forms that
/// the tensor expression operand could take:
/// - `R()`
/// - `R(ti_A, ti_a)`
/// - `(R(ti_A, ti_B) * S(ti_a, ti_b))`
///
/// \tparam T the derived TensorExpression type of the tensor expression operand
/// of the difference
/// \tparam X the type of data stored in the tensor expression operand of the
/// difference
/// \param t the tensor expression operand of the difference
/// \param number the `double` operand of the difference
/// \return the tensor expression representing the difference of a tensor
/// expression and a `double`
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>, tmpl::list<>>& t,
    const double number) {
  return t - TensorExpressions::NumberAsExpression(number);
}
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator-(
    const double number,
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>, tmpl::list<>>& t) {
  return TensorExpressions::NumberAsExpression(number) - t;
}
// @}
