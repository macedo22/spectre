// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor products

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"

namespace TensorExpressions {

namespace detail {

/*template <typename I1, typename I2>
using indices_contractible =
    std::integral_constant<
        bool,
        I1::dim == I2::dim and I1::ul != I2::ul and
            std::is_same_v<typename I1::Frame, typename I2::Frame> and
            I1::index_type == I2::index_type>;*/

template <typename T, typename X, typename SymmList, typename IndexList,
          typename Args>
struct ComputeContractedProductTypeImpl;

template <typename T, typename X, template <typename...> class SymmList,
          typename IndexList, typename Args, typename... Symm>
struct ComputeContractedProductTypeImpl<T, X, SymmList<Symm...>, IndexList,
                                        Args> {
  using type =
      TensorExpression<T, X, Symmetry<Symm::value...>, IndexList, Args>;
};

/*template <typename ReplacedArg1, typename ReplacedArg2, typename T, typename
X, typename Symm, typename IndexList, typename Args> using ComputeContractedType
= typename ComputeContractedProductTypeImpl< T, X, tmpl::erase<tmpl::erase<Symm,
tmpl::index_of< Args, TensorIndex<ReplacedArg2::value>>>, tmpl::index_of<Args,
TensorIndex<ReplacedArg1::value>>>, tmpl::erase< tmpl::erase<IndexList,
                    tmpl::index_of<Args, TensorIndex<ReplacedArg2::value>>>,
        tmpl::index_of<Args, TensorIndex<ReplacedArg1::value>>>,
    tmpl::erase<tmpl::erase<Args, tmpl::index_of<
                                      Args, TensorIndex<ReplacedArg2::value>>>,
                tmpl::index_of<Args,
TensorIndex<ReplacedArg1::value>>>>::type;*/

// this product type needs to so recursive replacements. TODO
template <typename ReplacedArg1, typename ReplacedArg2, typename T, typename X,
          typename Symm, typename IndexList, typename Args>
using ComputeContractedProductType = typename ComputeContractedProductTypeImpl<
    T, X,
    tmpl::erase<tmpl::erase<Symm, tmpl::index_of<
                                      Args, TensorIndex<ReplacedArg2::value>>>,
                tmpl::index_of<Args, TensorIndex<ReplacedArg1::value>>>,
    tmpl::erase<
        tmpl::erase<IndexList,
                    tmpl::index_of<Args, TensorIndex<ReplacedArg2::value>>>,
        tmpl::index_of<Args, TensorIndex<ReplacedArg1::value>>>,
    tmpl::erase<tmpl::erase<Args, tmpl::index_of<
                                      Args, TensorIndex<ReplacedArg2::value>>>,
                tmpl::index_of<Args, TensorIndex<ReplacedArg1::value>>>>::type;

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
 *
 * @tparam T1
 * @tparam T2
 * @tparam ArgsList1
 * @tparam ArgsList2
 */
template <typename T1, typename T2, typename ArgsList1, typename ArgsList2>
struct Product;

template <typename T1, typename T2, template <typename...> class ArgsList1,
          template <typename...> class ArgsList2, typename... Args1,
          typename... Args2>
struct Product<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>>
    : public TensorExpression<
          Product<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>>,
          typename T1::type, double,
          tmpl::append<typename T1::index_list, typename T2::index_list>,
          /*tmpl::sort<
              tmpl::append<typename T1::args_list, typename T2::args_list>>*/
          tmpl::append<typename T1::args_list, typename T2::args_list>> {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value,
                "Cannot product Tensors holding different data types.");
  // I think this is the largest element from the symmetry of T2
  using old_max_symm2 = tmpl::fold<typename T2::symmetry, tmpl::uint32_t<0>,
                                   tmpl::max<tmpl::_state, tmpl::_element>>;

  using type = typename T1::type;
  // I think this is adds max_symm2 to each of the symmetry elements for T1 and
  // then appends the symmetry of T2 onto this
  // e.g. If T1's symm was <2, 1, 2> and T2's symm was <2, 1, 1> then max_symm2
  // is 2, then T1 shifted up by 2 is <4, 3, 4> and then concatenated with T2's
  // symm would be: <4, 3, 4, 2, 1, 1>
  using combined_symmetry =
      tmpl::append<tmpl::transform<typename T1::symmetry,
                                   tmpl::plus<tmpl::_1, old_max_symm2>>,
                   typename T2::symmetry>;
  using combined_index_list =
      tmpl::append<typename T1::index_list, typename T2::index_list>;
  static constexpr auto combined_num_tensor_indices =
      tmpl::size<combined_index_list>::value;
  // tmpl::size<index_list>::value == 0 ? 1 : tmpl::size<index_list>::value;
  using combined_args_list =
      tmpl::append<typename T1::args_list, typename T2::args_list>;
  // tmpl::sort<tmpl::append<typename T1::args_list, typename T2::args_list>>;
  // using structure = Tensor_detail::Structure<symmetry, index_list>; TODO

  using repeated_combined_args_list =
      repeated<combined_args_list>;  // things to contract
  using replaced_repeated_combined_args_list =
      replace_indices<combined_args_list, repeated_combined_args_list>;

  using lhs_args = ArgsList1<Args1...> : using rhs_args = ArgsList1<Args2...> :

      Product(const T1& t1, const T2& t2)
      : t1_(t1),
        t2_(t2) {}

  // TODO: The args will need to be reduced in a careful manner, which means
  // they need to be reduced together, then split at the correct length so that
  // the indexing is correct.
  template <typename... LhsIndices, typename U>
  SPECTRE_ALWAYS_INLINE type
  get(const std::array<U, num_tensor_indices>& tensor_index) const {
    return t1_.template get<LhsIndices...>(tensor_index) *
           t2_.template get<LhsIndices...>(tensor_index);
  }

 private:
  const T1 t1_;
  const T2 t2_;
};

}  // namespace TensorExpressions

/*!
 * @ingroup TensorExpressionsGroup
 *
 * @tparam T1
 * @tparam T2
 * @tparam X
 * @tparam Symm1
 * @tparam Symm2
 * @tparam IndexList1
 * @tparam IndexList2
 * @tparam Args1
 * @tparam Args2
 * @param t1
 * @param t2
 * @return
 */
template <typename T1, typename T2, typename X, typename Symm1, typename Symm2,
          typename IndexList1, typename IndexList2, typename Args1,
          typename Args2>
SPECTRE_ALWAYS_INLINE auto operator*(
    const TensorExpression<T1, X, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X, Symm2, IndexList2, Args2>& t2) {
  // static_assert(tmpl::size<Args1>::value == tmpl::size<Args2>::value,
  //               "Tensor addition is only possible with the same rank
  //               tensors");
  // static_assert(tmpl::equal_members<Args1, Args2>::value,
  //               "The indices when adding two tensors must be equal. This
  //               error "
  //               "occurs from expressions like A(_a, _b) + B(_c, _a)");
  return TensorExpressions::Product<
      typename std::conditional<
          std::is_base_of<Expression, T1>::value, T1,
          TensorExpression<T1, X, Symm1, IndexList1, Args1>>::type,
      typename std::conditional<
          std::is_base_of<Expression, T2>::value, T2,
          TensorExpression<T2, X, Symm2, IndexList2, Args2>>::type,
      Args1, Args2>(~t1, ~t2);
}
