// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for adding and subtracting tensors

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>

#include "DataStructures/Tensor/Expressions/ScalarDataType.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
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
template <typename IndexList1, typename IndexList2, typename Args1,
          typename Args2, typename Element>
struct AddSubIndexCheckHelper
    : std::is_same<tmpl::at<IndexList1, tmpl::index_of<Args1, Element>>,
                   tmpl::at<IndexList2, tmpl::index_of<Args2, Element>>>::type {
};

// Check to make sure that the tensor indices being added are of the same type,
// dimensionality and in the same frame
template <typename IndexList1, typename IndexList2, typename Args1,
          typename Args2>
using AddSubIndexCheck = tmpl::fold<
    Args1, tmpl::bool_<true>,
    tmpl::and_<tmpl::_state,
               AddSubIndexCheckHelper<tmpl::pin<IndexList1>,
                                      tmpl::pin<IndexList2>, tmpl::pin<Args1>,
                                      tmpl::pin<Args2>, tmpl::_element>>>;
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
          typename T1::type,
          tmpl::transform<typename T1::symmetry, typename T2::symmetry,
                          tmpl::append<tmpl::max<tmpl::_1, tmpl::_2>>>,
          typename T1::index_list, tmpl::sort<typename T1::args_list>> {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value,
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

  using type = typename T1::type;
  using symmetry = tmpl::transform<typename T1::symmetry, typename T2::symmetry,
                                   tmpl::append<tmpl::max<tmpl::_1, tmpl::_2>>>;
  using index_list = typename T1::index_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  using args_list = tmpl::sort<typename T1::args_list>;

  AddSub(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}

  template <typename... LhsIndices, typename T>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<T, num_tensor_indices>& tensor_index) const {
    if constexpr (Sign == 1) {
      return t1_.template get<LhsIndices...>(tensor_index) +
             t2_.template get<LhsIndices...>(tensor_index);
    } else {
      return t1_.template get<LhsIndices...>(tensor_index) -
             t2_.template get<LhsIndices...>(tensor_index);
    }
  }

  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const size_t lhs_storage_index) const {
    if constexpr (Sign == 1) {
      return t1_.template get<LhsStructure, LhsIndices...>(lhs_storage_index) +
             t2_.template get<LhsStructure, LhsIndices...>(lhs_storage_index);
    } else {
      return t1_.template get<LhsStructure, LhsIndices...>(lhs_storage_index) -
             t2_.template get<LhsStructure, LhsIndices...>(lhs_storage_index);
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
  const T1 t1_;
  const T2 t2_;
};
}  // namespace TensorExpressions

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the sum of two tensor
/// expressions
///
/// \tparam T1 the derived TensorExpression type of the first operand of the sum
/// \tparam T2 the derived TensorExpression type of the second operand of the
/// sum
/// \tparam Symm1 the Symmetry of the first operand
/// \tparam Symm2 the Symmetry of the second operand
/// \tparam IndexList1 the \ref SpacetimeIndex "TensorIndexType"s of the first
/// operand
/// \tparam IndexList2 the \ref SpacetimeIndex "TensorIndexType"s of the second
/// operand
/// \tparam ArgsList1 the TensorIndexs of the first operand
/// \tparam ArgsList2 the TensorIndexs of the second operand
/// \param t1 first operand expression of the sum
/// \param t2 second operand expression of the sum
/// \return the tensor expression representing the sum of two tensor expressions
template <typename T1, typename T2, typename X, typename Symm1, typename Symm2,
          typename IndexList1, typename IndexList2, typename ArgsList1,
          typename ArgsList2>
SPECTRE_ALWAYS_INLINE auto operator+(
    const TensorExpression<T1, X, Symm1, IndexList1, ArgsList1>& t1,
    const TensorExpression<T2, X, Symm2, IndexList2, ArgsList2>& t2) {
  static_assert(tmpl::size<ArgsList1>::value == tmpl::size<ArgsList2>::value,
                "Tensor addition is only possible with the same rank tensors");
  static_assert(tmpl::equal_members<ArgsList1, ArgsList2>::value,
                "The indices when adding two tensors must be equal. This error "
                "occurs from expressions like A(_a, _b) + B(_c, _a)");
  return TensorExpressions::AddSub<
      tmpl::conditional_t<
          std::is_base_of<Expression, T1>::value, T1,
          TensorExpression<T1, X, Symm1, IndexList1, ArgsList1>>,
      tmpl::conditional_t<
          std::is_base_of<Expression, T2>::value, T2,
          TensorExpression<T2, X, Symm2, IndexList2, ArgsList2>>,
      ArgsList1, ArgsList2, 1>(~t1, ~t2);
}  // R(ti_a) + T(ti_b)
// delete const reference, delete non-const reference version of above?
// take t1 and t2 by value and move?
// take t1 and t2 by rvalue? (and move)

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the sum of a tensor
/// expression and a `double` or DataVector
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
/// There are four separate overloads for the sum of a tensor expression and a
/// a `double` or DataVector, where this function is one of them:
/// - `operator+(const X& scalar, const Expression& t)`
/// - `operator+(const Expression& t, const X& scalar)`
/// - `operator+(X&& scalar, const Expression& t)`
/// - `operator+(const Expression& t, X&& scalar)`
///
/// The last two overloads are necessary so that DataVector r-values can be
/// moved to an expression instead of the expression pointing to an object that
/// will go out of scope.
///
/// \tparam T the derived TensorExpression type of the tensor expression operand
/// of the sum
/// \tparam X the data type of the operands, `double` or DataVector
/// \param scalar the scalar type operand of the sum, a `double` or DataVector
/// \param t the expression operand of the sum
/// \return the tensor expression representing the sum of a tensor expression
/// and a `double` or DataVector
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator+(
    const X& scalar,
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>>& t) {
  return TensorExpressions::ScalarDataType<X, false>(scalar) + t;
}

/// \ingroup TensorExpressionsGroup
/// \copydoc operator+(const X&,const TensorExpression<T, X, tmpl::list<>,
/// tmpl::list<>>&)
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator+(
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>>& t,
    const X& scalar) {
  // std::cout << "lvalue" << std::endl;
  return t + TensorExpressions::ScalarDataType<X, false>(scalar);
}

/// \ingroup TensorExpressionsGroup
/// \copydoc operator+(const X&,const TensorExpression<T, X, tmpl::list<>,
/// tmpl::list<>>&)
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator+(
    X&& scalar, const TensorExpression<T, X, tmpl::list<>, tmpl::list<>>& t) {
  return TensorExpressions::ScalarDataType<X, true>(std::move(scalar)) + t;
}

/// \ingroup TensorExpressionsGroup
/// \copydoc operator+(const X&,const TensorExpression<T, X, tmpl::list<>,
/// tmpl::list<>>&)
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator+(
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>>& t, X&& scalar) {
  // std::cout << "t : " << t << std::endl;
  // std::cout << "scalar : " << scalar << std::endl;
  // std::cout << "rvalue" << std::endl;
  return t + TensorExpressions::ScalarDataType<X, true>(std::move(scalar));
}

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the difference of two
/// tensor expressions
///
/// \tparam T1 the derived TensorExpression type of the first operand of the
/// difference
/// \tparam T2 the derived TensorExpression type of the second operand of the
/// difference
/// \tparam Symm1 the Symmetry of the first operand
/// \tparam Symm2 the Symmetry of the second operand
/// \tparam IndexList1 the \ref SpacetimeIndex "TensorIndexType"s of the first
/// operand
/// \tparam IndexList2 the \ref SpacetimeIndex "TensorIndexType"s of the second
/// operand
/// \tparam ArgsList1 the TensorIndexs of the first operand
/// \tparam ArgsList2 the TensorIndexs of the second operand
/// \param t1 first operand expression of the difference
/// \param t2 second operand expression of the difference
/// \return the tensor expression representing the difference of two tensor
/// expressions
template <typename T1, typename T2, typename X, typename Symm1, typename Symm2,
          typename IndexList1, typename IndexList2, typename ArgsList1,
          typename ArgsList2>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T1, X, Symm1, IndexList1, ArgsList1>& t1,
    const TensorExpression<T2, X, Symm2, IndexList2, ArgsList2>& t2) {
  static_assert(
      tmpl::size<ArgsList1>::value == tmpl::size<ArgsList2>::value,
      "Tensor subtraction is only possible with the same rank tensors");
  static_assert(tmpl::equal_members<ArgsList1, ArgsList2>::value,
                "The indices when subtracting two tensors must be equal. This "
                "error occurs from expressions like A(_a, _b) - B(_c, _a)");
  return TensorExpressions::AddSub<
      tmpl::conditional_t<
          std::is_base_of<Expression, T1>::value, T1,
          TensorExpression<T1, X, Symm1, IndexList1, ArgsList1>>,
      tmpl::conditional_t<
          std::is_base_of<Expression, T2>::value, T2,
          TensorExpression<T2, X, Symm2, IndexList2, ArgsList2>>,
      ArgsList1, ArgsList2, -1>(~t1, ~t2);
}

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the difference of a tensor
/// expression and a `double` or DataVector
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
/// There are four separate overloads for the difference of a tensor expression
/// and a `double` or DataVector, where this function is one of them:
/// - `operator-(const X& scalar, const Expression& t)`
/// - `operator-(const Expression& t, const X& scalar)`
/// - `operator-(X&& scalar, const Expression& t)`
/// - `operator-(const Expression& t, X&& scalar)`
///
/// The last two overloads are necessary so that DataVector r-values can be
/// moved to an expression instead of the expression pointing to an object that
/// will go out of scope.
///
/// \tparam T the derived TensorExpression type of the tensor expression operand
/// of the difference
/// \tparam X the data type of the operands, `double` or DataVector
/// \param scalar the scalar type operand of the difference, a `double` or
/// DataVector
/// \param t the expression operand of the difference
/// \return the tensor expression representing the difference of a tensor
/// expression and a `double` or DataVector
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator-(
    const X& scalar,
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>>& t) {
  return TensorExpressions::ScalarDataType<X, false>(scalar) - t;
}

/// \ingroup TensorExpressionsGroup
/// \copydoc operator-(const X&,const TensorExpression<T, X, tmpl::list<>,
/// tmpl::list<>>&)
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>>& t,
    const X& scalar) {
  return t - TensorExpressions::ScalarDataType<X, false>(scalar);
}

/// \ingroup TensorExpressionsGroup
/// \copydoc operator-(const X&,const TensorExpression<T, X, tmpl::list<>,
/// tmpl::list<>>&)
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator-(
    X&& scalar, const TensorExpression<T, X, tmpl::list<>, tmpl::list<>>& t) {
  return TensorExpressions::ScalarDataType<X, true>(std::move(scalar)) - t;
}

/// \ingroup TensorExpressionsGroup
/// \copydoc operator-(const X&,const TensorExpression<T, X, tmpl::list<>,
/// tmpl::list<>>&)
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>>& t, X&& scalar) {
  return t - TensorExpressions::ScalarDataType<X, true>(std::move(scalar));
}
