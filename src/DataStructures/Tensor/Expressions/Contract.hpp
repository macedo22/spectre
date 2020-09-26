// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Expression Templates for contracting tensor indices on a single
/// Tensor

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
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
  using Index1 = tmpl::int32_t<
      tmpl::index_of<ArgsList, TensorIndex<ReplacedArg1::value>>::value>;
  using Index2 = tmpl::int32_t<
      tmpl::index_of<ArgsList, TensorIndex<ReplacedArg2::value>>::value>;
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
  using args_list = tmpl::sort<typename new_type::args_list>;

  explicit TensorContract(
      const TensorExpression<T, X, Symm, IndexList, ArgsList>& t)
      : t_(~t) {}

  // Returning constexpr isn't working
  /*template <size_t Rank>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, Rank>
  fill_contracting_tensor_index(
      // std::array<size_t, Rank>& tensor_index_in,
      const std::array<size_t, num_tensor_indices>& tensor_index) {
    std::array<size_t, Rank> tensor_index_in;

    // fill all I before Index1
    cpp20::copy(tensor_index.begin(), tensor_index.begin() + Index1::value,
                tensor_index_in.begin());
    // fill I == Index1
    tensor_index_in[Index1::value] = 10000;

    // fill between Index1 and Index2
    cpp20::copy(tensor_index.begin() + Index1::value,
                tensor_index.begin() + Index2::value,
                tensor_index_in.begin() + Index1::value + 1);

    // if we're after Index2 but not at the last element
    if constexpr (Index2::value + 1 < Rank - 1) {
      cpp20::copy(tensor_index.begin() + Index2::value - 1,
                  tensor_index.end() - 1,
                  tensor_index_in.begin() + Index2::value + 1);
    }

    // Handle last element
    // if Index2 is the last I
    if constexpr (Index2::value == (Rank - 1)) {
      tensor_index_in[Rank - 1] = 20000;
    } else {
      tensor_index_in[Index2::value] = 20000;
      tensor_index_in[Rank - 1] = tensor_index[Rank - 3];
    }
    return tensor_index_in;
  }*/

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

  template <size_t Rank>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, Rank> test_copy(
      const std::array<size_t, num_tensor_indices>& to_copy) {
    std::array<size_t, Rank> test;
    cpp20::copy(to_copy.begin(), to_copy.end(), test.begin());
    return test;
  }

  template <typename... LhsIndices, typename U>
  SPECTRE_ALWAYS_INLINE type
  get(const std::array<U, num_tensor_indices>& new_tensor_index) const {
    // new_tensor_index is the one with _fewer_ components, ie post-contraction
    std::array<size_t, tmpl::size<Symm>::value> tensor_index;
    // Manually unrolled for loops to compute the tensor_index from the
    // new_tensor_index
    fill_contracting_tensor_index<0>(tensor_index, new_tensor_index);

    // tensor_index isn't being assigned to constexpr return value
    /*constexpr std::array<size_t, tmpl::size<Symm>::value> tensor_index =
        fill_contracting_tensor_index<tmpl::size<Symm>::value>(
            new_tensor_index);*/

    // this is to test constexpr std::copy (i.e. cpp20::copy) :
    // copy_returned isn't being assigned to constexpr return value
    constexpr std::array<size_t, tmpl::size<Symm>::value> copy_returned =
        test_copy<tmpl::size<Symm>::value>(new_tensor_index);

    return detail::ComputeContractionImpl<CI1::dim - 1, Index1, Index2>::
        template apply<LhsIndices...>(tensor_index, t_);
  }

 private:
  const std::conditional_t<std::is_base_of<Expression, T>::value, T,
                           TensorExpression<T, X, Symm, IndexList, ArgsList>>
      t_;
};

/*!
 * \ingroup TensorExpressionsGroup
 */
template <int ReplacedArg1, int ReplacedArg2, typename T, typename X,
          typename Symm, typename IndexList, typename Args>
SPECTRE_ALWAYS_INLINE auto contract(
    const TensorExpression<T, X, Symm, IndexList, Args>& t) {
  return TensorContract<tmpl::int32_t<ReplacedArg1>,
                        tmpl::int32_t<ReplacedArg2>, T, X, Symm, IndexList,
                        Args>(~t);
}

namespace detail {
// Helper struct to allow contractions by using repeated indices in operator()
// calls to tensor.
template <template <typename> class TE, typename ReplacedArgList, typename I,
          typename TotalContracted>
struct fully_contract_helper {
  template <typename T>
  SPECTRE_ALWAYS_INLINE static constexpr auto apply(
      const T& t) -> decltype(contract<ti_contracted_t<I::value>::value,
                                       ti_contracted_t<I::value + 1>::value>(
      fully_contract_helper<TE, ReplacedArgList, tmpl::int32_t<I::value + 1>,
                            TotalContracted>::apply(t))) {
    return contract<ti_contracted_t<I::value>::value,
                    ti_contracted_t<I::value + 1>::value>(
        fully_contract_helper<TE, ReplacedArgList, tmpl::int32_t<I::value + 1>,
                              TotalContracted>::apply(t));
  }
};

template <template <typename> class TE, typename ReplacedArgList,
          typename TotalContracted>
struct fully_contract_helper<TE, ReplacedArgList,
                             tmpl::int32_t<TotalContracted::value - 1>,
                             TotalContracted> {
  using I = tmpl::int32_t<2 * (TotalContracted::value - 1)>;
  template <typename T>
  SPECTRE_ALWAYS_INLINE static constexpr auto apply(const T& t) -> decltype(
      contract<ti_contracted_t<I::value>::value,
               ti_contracted_t<I::value + 1>::value>(TE<ReplacedArgList>(t))) {
    return contract<ti_contracted_t<I::value>::value,
                    ti_contracted_t<I::value + 1>::value>(
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
