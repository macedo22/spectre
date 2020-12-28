// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor products

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
namespace detail {
template <typename T1, typename T2, typename SymmList1 = typename T1::symmetry,
          typename SymmList2 = typename T2::symmetry>
struct ProductType;

template <typename T1, typename T2, template <typename...> class SymmList1,
          typename... Symm1, template <typename...> class SymmList2,
          typename... Symm2>
struct ProductType<T1, T2, SymmList1<Symm1...>, SymmList2<Symm2...>> {
  using symmetry =
      Symmetry<(Symm1::value + sizeof...(Symm2))..., Symm2::value...>;
  using index_list =
      tmpl::append<typename T1::index_list, typename T2::index_list>;
  using tensorindex_list =
      tmpl::append<typename T1::args_list, typename T2::args_list>;
};

template <typename LhsTensorIndexList, typename OperandRhsTensorIndexList>
using get_operand_lhs_tensorindex_list = tmpl::filter<
    LhsTensorIndexList,
    tmpl::bind<tmpl::found, tmpl::pin<OperandRhsTensorIndexList>,
               tmpl::bind<std::is_same, tmpl::_1, tmpl::parent<tmpl::_1>>>>;
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 *
 * @tparam T1 eh
 * @tparam T2 eh
 * @tparam ArgsList1 eh
 * @tparam ArgsList2 eh
 */
template <typename T1, typename T2, typename ArgsList1, typename ArgsList2>
struct Product;

template <typename T1, typename T2, template <typename...> class ArgsList1,
          template <typename...> class ArgsList2, typename... Args1,
          typename... Args2>
struct Product<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>>
    : public TensorExpression<
          Product<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>>,
          typename T1::type, typename detail::ProductType<T1, T2>::symmetry,
          typename detail::ProductType<T1, T2>::index_list,
          typename detail::ProductType<T1, T2>::tensorindex_list> {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value,
                "Cannot product Tensors holding different data types.");
  using type = typename T1::type;
  using symmetry = typename detail::ProductType<T1, T2>::symmetry;
  using index_list = typename detail::ProductType<T1, T2>::index_list;
  using args_list = typename detail::ProductType<T1, T2>::tensorindex_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  static constexpr auto num_tensor_indices_first_operand =
      tmpl::size<typename T1::index_list>::value;
  static constexpr auto num_tensor_indices_second_operand =
      num_tensor_indices - num_tensor_indices_first_operand;

  Product(const T1& t1, const T2& t2)
      : t1_(std::move(t1)), t2_(std::move(t2)) {}

  template <typename OperandLhsTensorIndexList>
  struct GetOperandTensorMultiIndex;

  template <typename... OperandLhsTensorIndices>
  struct GetOperandTensorMultiIndex<tmpl::list<OperandLhsTensorIndices...>> {
    template <typename... LhsTensorIndices>
    static SPECTRE_ALWAYS_INLINE constexpr std::array<
        size_t, sizeof...(OperandLhsTensorIndices)>
    apply(
        const std::array<size_t, num_tensor_indices>& lhs_tensor_multi_index) {
      constexpr size_t operand_num_tensor_indices =
          sizeof...(OperandLhsTensorIndices);
      // e.g. <ti_c, ti_B, ti_b, ti_a, ti_A, ti_d>
      constexpr std::array<size_t, sizeof...(LhsTensorIndices)>
          lhs_tensorindex_vals = {{LhsTensorIndices::value...}};
      // e.g. <ti_A, ti_b, ti_c>
      constexpr std::array<size_t, operand_num_tensor_indices>
          operand_lhs_tensorindex_vals = {{OperandLhsTensorIndices::value...}};
      // to fill
      std::array<size_t, operand_num_tensor_indices>
          operand_lhs_tensor_multi_index;

      for (size_t i = 0; i < operand_num_tensor_indices; i++) {
        gsl::at(operand_lhs_tensor_multi_index, i) =
            gsl::at(lhs_tensor_multi_index,
                    static_cast<unsigned long>(std::distance(
                        lhs_tensorindex_vals.begin(),
                        alg::find(lhs_tensorindex_vals,
                                  gsl::at(operand_lhs_tensorindex_vals, i)))));
      }
      return operand_lhs_tensor_multi_index;
    }
  };

  template <typename FirstOperandLhsTensorIndexList,
            typename SecondOperandLhsTensorIndexList>
  struct ComputeProduct;

  template <typename... FirstOperandLhsTensorIndices,
            typename... SecondOperandLhsTensorIndices>
  struct ComputeProduct<tmpl::list<FirstOperandLhsTensorIndices...>,
                        tmpl::list<SecondOperandLhsTensorIndices...>> {
    template <typename UncontractedLhsStructureFirstOp,
              typename UncontractedLhsStructureSecondOp>
    static SPECTRE_ALWAYS_INLINE decltype(auto) apply(
        const size_t first_storage_index_operand,
        const size_t second_storage_index_operand, const T1& t1, const T2& t2) {
      return t1.template get<UncontractedLhsStructureFirstOp,
                             FirstOperandLhsTensorIndices...>(
                 first_storage_index_operand) *
             t2.template get<UncontractedLhsStructureSecondOp,
                             SecondOperandLhsTensorIndices...>(
                 second_storage_index_operand);
    }
  };

  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const size_t lhs_storage_index) const {
    const std::array<size_t, num_tensor_indices>& lhs_tensor_multi_index =
        LhsStructure::get_canonical_tensor_index(lhs_storage_index);

    using first_op_tensorindex_list =
        detail::get_operand_lhs_tensorindex_list<tmpl::list<LhsIndices...>,
                                                 ArgsList1<Args1...>>;

    using second_op_tensorindex_list =
        detail::get_operand_lhs_tensorindex_list<tmpl::list<LhsIndices...>,
                                                 ArgsList2<Args2...>>;

    std::array<size_t, num_tensor_indices_first_operand>
        first_tensor_index_operand =
            GetOperandTensorMultiIndex<first_op_tensorindex_list>::
                template apply<LhsIndices...>(lhs_tensor_multi_index);

    std::array<size_t, num_tensor_indices_second_operand>
        second_tensor_index_operand =
            GetOperandTensorMultiIndex<second_op_tensorindex_list>::
                template apply<LhsIndices...>(lhs_tensor_multi_index);

    using uncontracted_lhs_structure_first_op =
        typename LhsTensorSymmAndIndices<
            ArgsList1<Args1...>, first_op_tensorindex_list,
            typename T1::symmetry, typename T1::index_list>::structure;
    const size_t first_storage_index_operand =
        uncontracted_lhs_structure_first_op::get_storage_index(
            first_tensor_index_operand);
    using uncontracted_lhs_structure_second_op =
        typename LhsTensorSymmAndIndices<
            ArgsList2<Args2...>, second_op_tensorindex_list,
            typename T2::symmetry, typename T2::index_list>::structure;
    const size_t second_storage_index_operand =
        uncontracted_lhs_structure_second_op::get_storage_index(
            second_tensor_index_operand);

    return ComputeProduct<first_op_tensorindex_list,
                          second_op_tensorindex_list>::
        template apply<uncontracted_lhs_structure_first_op,
                       uncontracted_lhs_structure_second_op>(
            first_storage_index_operand, second_storage_index_operand, t1_,
            t2_);
  }

 private:
  const T1 t1_;
  const T2 t2_;
};

}  // namespace TensorExpressions

/*!
 * @ingroup TensorExpressionsGroup
 *
 * @tparam T1 eh
 * @tparam T2 eh
 * @tparam X eh
 * @tparam Symm1 eh
 * @tparam Symm2 eh
 * @tparam IndexList1 eh
 * @tparam IndexList2 eh
 * @tparam Args1 eh
 * @tparam Args2 eh
 * @param t1 eh
 * @param t2 eh
 * @return eh
 */
template <typename T1, typename T2, typename X, typename Symm1, typename Symm2,
          typename IndexList1, typename IndexList2, typename Args1,
          typename Args2>
SPECTRE_ALWAYS_INLINE auto operator*(
    const TensorExpression<T1, X, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X, Symm2, IndexList2, Args2>& t2) {
  return TensorExpressions::contract(
      TensorExpressions::Product<
          typename std::conditional<
              std::is_base_of<Expression, T1>::value, T1,
              TensorExpression<T1, X, Symm1, IndexList1, Args1>>::type,
          typename std::conditional<
              std::is_base_of<Expression, T2>::value, T2,
              TensorExpression<T2, X, Symm2, IndexList2, Args2>>::type,
          Args1, Args2>(~t1, ~t2));
}
