// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor inner and outer products

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
namespace detail {
template <typename T1, typename T2, typename SymmList1 = typename T1::symmetry,
          typename SymmList2 = typename T2::symmetry>
struct OuterProductType;

template <typename T1, typename T2, template <typename...> class SymmList1,
          typename... Symm1, template <typename...> class SymmList2,
          typename... Symm2>
struct OuterProductType<T1, T2, SymmList1<Symm1...>, SymmList2<Symm2...>> {
  using symmetry =
      Symmetry<(Symm1::value + sizeof...(Symm2))..., Symm2::value...>;
  using index_list =
      tmpl::append<typename T1::index_list, typename T2::index_list>;
  using tensorindex_list =
      tmpl::append<typename T1::args_list, typename T2::args_list>;
};
}  // namespace detail

/// \ingroup TensorExpressionsGroup
///
/// \tparam T1 eh
/// \tparam T2 eh
/// \tparam ArgsList1 eh
/// \tparam ArgsList2 eh
template <typename T1, typename T2,
          typename IndexList1 = typename T1::index_list,
          typename ArgsList1 = typename T1::args_list,
          typename IndexList2 = typename T2::index_list,
          typename ArgsList2 = typename T2::args_list>
struct OuterProduct;

template <typename T1, typename T2, template <typename...> class IndexList1,
          typename... Indices1, template <typename...> class ArgsList1,
          typename... Args1, template <typename...> class IndexList2,
          typename... Indices2, template <typename...> class ArgsList2,
          typename... Args2>
struct OuterProduct<T1, T2, IndexList1<Indices1...>, ArgsList1<Args1...>,
                    IndexList2<Indices2...>, ArgsList2<Args2...>>
    : public TensorExpression<
          OuterProduct<T1, T2, IndexList1<Indices1...>, ArgsList1<Args1...>,
                       IndexList2<Indices2...>, ArgsList2<Args2...>>,
          typename T1::type,
          typename detail::OuterProductType<T1, T2>::symmetry,
          typename detail::OuterProductType<T1, T2>::index_list,
          typename detail::OuterProductType<T1, T2>::tensorindex_list> {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value,
                "Cannot product Tensors holding different data types.");

  using type = typename T1::type;
  using symmetry = typename detail::OuterProductType<T1, T2>::symmetry;
  using index_list = typename detail::OuterProductType<T1, T2>::index_list;
  using args_list = typename detail::OuterProductType<T1, T2>::tensorindex_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;

  using first_op_structure =
      Tensor_detail::Structure<typename T1::symmetry, Indices1...>;
  using second_op_structure =
      Tensor_detail::Structure<typename T2::symmetry, Indices2...>;
  static constexpr auto num_tensor_indices_first_operand =
      tmpl::size<typename T1::index_list>::value;
  static constexpr auto num_tensor_indices_second_operand =
      num_tensor_indices - num_tensor_indices_first_operand;

  OuterProduct(const T1& t1, const T2& t2)
      : t1_(std::move(t1)), t2_(std::move(t2)) {}

  /// \brief Get a product expression operand's generic indices in the order
  /// they appear in the generic indices of the LHS outer product
  ///
  /// \details
  /// `LhsTensorIndexList` represents the list of generic indices of the outer
  /// product of the two operands from the RHS product expression, potentially
  /// reordered on the LHS. `OperandTensorIndexList` represents one such
  /// operand's generic indices in the order they appear in the RHS expression.
  ///
  /// Example: Let `ti_a_t` denote the type of `ti_a`, and apply the same
  /// convention for other generic indices. If `LhsTensorIndexList` is
  /// <ti_a_t, ti_A_t, ti_c_t, ti_b_t> and `OperandTensorIndexList` is
  /// <ti_b_t, ti_A_t>, then this alias will evaluate to <ti_A_t, ti_b_t>.
  ///
  /// \tparam LhsTensorIndexList the list of TensorIndexs of the outer product
  /// on the LHS
  /// \tparam OperandTensorIndexList the list of TensorIndexs of an operand in
  /// the RHS expression
  template <typename LhsTensorIndexList, typename OperandTensorIndexList>
  using get_operand_lhs_tensorindex_list = tmpl::filter<
      LhsTensorIndexList,
      tmpl::bind<tmpl::found, tmpl::pin<OperandTensorIndexList>,
                 tmpl::bind<std::is_same, tmpl::_1, tmpl::parent<tmpl::_1>>>>;

  template <typename OperandTensorIndexList>
  struct GetOpTensorMultiIndex;

  template <typename... OperandTensorIndices>
  struct GetOpTensorMultiIndex<tmpl::list<OperandTensorIndices...>> {
    template <typename... LhsTensorIndices>
    static SPECTRE_ALWAYS_INLINE constexpr std::array<
        size_t, sizeof...(OperandTensorIndices)>
    apply(
        const std::array<size_t, num_tensor_indices>& lhs_tensor_multi_index) {
      constexpr size_t operand_num_tensor_indices =
          sizeof...(OperandTensorIndices);
      // e.g. <ti_c, ti_B, ti_b, ti_a, ti_A, ti_d>
      constexpr std::array<size_t, sizeof...(LhsTensorIndices)>
          lhs_tensorindex_vals = {{LhsTensorIndices::value...}};
      // e.g. <ti_A, ti_b, ti_c>
      constexpr std::array<size_t, operand_num_tensor_indices>
          operand_tensorindex_vals = {{OperandTensorIndices::value...}};
      // to fill
      std::array<size_t, operand_num_tensor_indices> operand_tensor_multi_index;

      for (size_t i = 0; i < operand_num_tensor_indices; i++) {
        gsl::at(operand_tensor_multi_index, i) =
            gsl::at(lhs_tensor_multi_index,
                    static_cast<unsigned long>(std::distance(
                        lhs_tensorindex_vals.begin(),
                        alg::find(lhs_tensorindex_vals,
                                  gsl::at(operand_tensorindex_vals, i)))));
      }
      return operand_tensor_multi_index;
    }
  };

  /// \brief Helper struct for computing a component of the LHS outer product
  ///
  /// \tparam FirstOpLhsTensorIndexList the list of TensorIndexs of the first
  /// operand in the RHS expression
  /// \tparam SecondOpLhsTensorIndexList the list of TensorIndexs of the second
  /// operand in the RHS expression
  template <typename FirstOpLhsTensorIndexList,
            typename SecondOpLhsTensorIndexList>
  struct ComputeOuterProductComponent;

  template <typename... FirstOpLhsTensorIndices,
            typename... SecondOpLhsTensorIndices>
  struct ComputeOuterProductComponent<tmpl::list<FirstOpLhsTensorIndices...>,
                                      tmpl::list<SecondOpLhsTensorIndices...>> {
    /// \brief Computes the value of a component in the LHS outer product
    template <typename FirstOpLhsStructure, typename SecondOpLhsStructure>
    static SPECTRE_ALWAYS_INLINE decltype(auto) apply(
        const size_t first_op_lhs_storage_index,
        const size_t second_op_lhs_storage_index, const T1& t1, const T2& t2) {
      return t1.template get<FirstOpLhsStructure, FirstOpLhsTensorIndices...>(
                 first_op_lhs_storage_index) *
             t2.template get<SecondOpLhsStructure, SecondOpLhsTensorIndices...>(
                 second_op_lhs_storage_index);
    }
  };

  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const size_t lhs_storage_index) const {
    const std::array<size_t, num_tensor_indices>& lhs_tensor_multi_index =
        LhsStructure::get_canonical_tensor_index(lhs_storage_index);

    const std::array<size_t, num_tensor_indices_first_operand>
        first_op_tensor_multi_index =
            GetOpTensorMultiIndex<ArgsList1<Args1...>>::template apply<
                LhsIndices...>(lhs_tensor_multi_index);
    const std::array<size_t, num_tensor_indices_second_operand>
        second_op_tensor_multi_index =
            GetOpTensorMultiIndex<ArgsList2<Args2...>>::template apply<
                LhsIndices...>(lhs_tensor_multi_index);

    const size_t first_op_storage_index =
        first_op_structure::get_storage_index(first_op_tensor_multi_index);
    const size_t second_op_storage_index =
        second_op_structure::get_storage_index(second_op_tensor_multi_index);

    return ComputeOuterProductComponent<ArgsList1<Args1...>,
                                        ArgsList2<Args2...>>::
        template apply<first_op_structure, second_op_structure>(
            first_op_storage_index, second_op_storage_index, t1_, t2_);
  }

 private:
  const T1 t1_;
  const T2 t2_;
};

}  // namespace TensorExpressions

/// \ingroup TensorExpressionsGroup
///
/// \tparam T1 eh
/// \tparam T2 eh
/// \tparam X eh
/// \tparam Symm1 eh
/// \tparam Symm2 eh
/// \tparam IndexList1 eh
/// \tparam IndexList2 eh
/// \tparam Args1 eh
/// \tparam Args2 eh
/// \param t1 eh
/// \param t2 eh
/// \return eh
template <typename T1, typename T2, typename X, typename Symm1, typename Symm2,
          typename IndexList1, typename IndexList2, typename Args1,
          typename Args2>
SPECTRE_ALWAYS_INLINE auto operator*(
    const TensorExpression<T1, X, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X, Symm2, IndexList2, Args2>& t2) {
  return TensorExpressions::contract(
      TensorExpressions::OuterProduct<
          typename std::conditional<
              std::is_base_of<Expression, T1>::value, T1,
              TensorExpression<T1, X, Symm1, IndexList1, Args1>>::type,
          typename std::conditional<
              std::is_base_of<Expression, T2>::value, T2,
              TensorExpression<T2, X, Symm2, IndexList2, Args2>>::type>(~t1,
                                                                        ~t2));
}
