// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor products

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
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

  // not sure if std::move is necessary, but it's used in AddSub constructor
  Product(const T1& t1, const T2& t2)
      : t1_(std::move(t1)), t2_(std::move(t2)) {}

  template <typename FirstOperandLhsTensorIndexList>
  struct GetFirstTensorMultiIndexOperand;

  template <typename... FirstOperandLhsTensorIndices>
  struct GetFirstTensorMultiIndexOperand<
      tmpl::list<FirstOperandLhsTensorIndices...>> {
    template <typename... LhsIndices>
    static SPECTRE_ALWAYS_INLINE constexpr std::array<
        size_t, num_tensor_indices_first_operand>
    apply(
        const std::array<size_t, num_tensor_indices>& lhs_tensor_multi_index) {
    //   std::cout << "=== PRODUCT GetFirstTensorMultiIndexOperand === "
    //             << std::endl;
    //   std::cout << "lhs_tensor_multi_index: " << lhs_tensor_multi_index
    //             << std::endl;
      // e.g. <ti_c, ti_B, ti_b, ti_a, ti_A, ti_d>
      constexpr std::array<size_t, sizeof...(LhsIndices)> lhs_tensorindex_vals =
          {{LhsIndices::value...}};
    //   std::cout << "lhs_tensorindex_vals: " << lhs_tensorindex_vals
    //             << std::endl;
      // e.g. <ti_A, ti_b, ti_c>
      constexpr std::array<size_t, num_tensor_indices_first_operand>
          first_op_tensorindex_vals = {
              {FirstOperandLhsTensorIndices::value...}};
    //   std::cout << "first_op_tensorindex_vals: " << first_op_tensorindex_vals
    //             << std::endl;
      // to fill
      std::array<size_t, num_tensor_indices_first_operand>
          first_lhs_tensor_multi_index_operand;

      for (size_t i = 0; i < num_tensor_indices_first_operand; i++) {
        // std::cout << "i: " << i << std::endl;
        // std::cout << "gsl::at(first_op_tensorindex_vals, i): "
        //           << gsl::at(first_op_tensorindex_vals, i) << std::endl;
        // std::cout << "pos of gsl::at(first_op_tensorindex_vals, i) in "
        //              "lhs_tensorindex_vals: "
        //           << static_cast<unsigned long>(std::distance(
        //                  lhs_tensorindex_vals.begin(),
        //                  alg::find(lhs_tensorindex_vals,
        //                            gsl::at(first_op_tensorindex_vals, i))))
        //           << std::endl;
        // std::cout << "value being assigned at "
        //              "first_lhs_tensor_multi_index_operand[i] : "
        //           << gsl::at(
        //                  lhs_tensor_multi_index,
        //                  static_cast<unsigned long>(std::distance(
        //                      lhs_tensorindex_vals.begin(),
        //                      alg::find(lhs_tensorindex_vals,
        //                                gsl::at(first_op_tensorindex_vals,
        //                                i)))))
        //           << std::endl;
        first_lhs_tensor_multi_index_operand[i] =
            gsl::at(lhs_tensor_multi_index,
                    static_cast<unsigned long>(std::distance(
                        lhs_tensorindex_vals.begin(),
                        alg::find(lhs_tensorindex_vals,
                                  gsl::at(first_op_tensorindex_vals, i)))));
      }
    //   std::cout << "returning first_lhs_tensor_multi_index_operand : "
    //             << first_lhs_tensor_multi_index_operand << std::endl;
      return first_lhs_tensor_multi_index_operand;
    }
  };

  template <typename SecondOperandLhsTensorIndexList>
  struct GetSecondTensorMultiIndexOperand;

  template <typename... SecondOperandLhsTensorIndices>
  struct GetSecondTensorMultiIndexOperand<
      tmpl::list<SecondOperandLhsTensorIndices...>> {
    template <typename... LhsIndices>
    static SPECTRE_ALWAYS_INLINE constexpr std::array<
        size_t, num_tensor_indices_second_operand>
    apply(
        const std::array<size_t, num_tensor_indices>& lhs_tensor_multi_index) {
    //   std::cout << "=== PRODUCT GetSecondTensorMultiIndexOperand === "
    //             << std::endl;
    //   std::cout << "lhs_tensor_multi_index: " << lhs_tensor_multi_index
    //             << std::endl;
      // e.g. <ti_c, ti_B, ti_b, ti_a, ti_A, ti_d>
      constexpr std::array<size_t, sizeof...(LhsIndices)> lhs_tensorindex_vals =
          {{LhsIndices::value...}};
    //   std::cout << "lhs_tensorindex_vals: " << lhs_tensorindex_vals
    //             << std::endl;
      // e.g. <ti_A, ti_b, ti_c>
      constexpr std::array<size_t, num_tensor_indices_second_operand>
          second_op_tensorindex_vals = {
              {SecondOperandLhsTensorIndices::value...}};
    //   std::cout << "second_op_tensorindex_vals: "
    //             << second_op_tensorindex_vals
    //             << std::endl;
      // to fill
      std::array<size_t, num_tensor_indices_second_operand>
          second_lhs_tensor_multi_index_operand;

      for (size_t i = 0; i < num_tensor_indices_second_operand; i++) {
        second_lhs_tensor_multi_index_operand[i] =
            gsl::at(lhs_tensor_multi_index,
                    static_cast<unsigned long>(std::distance(
                        lhs_tensorindex_vals.begin(),
                        alg::find(lhs_tensorindex_vals,
                                  gsl::at(second_op_tensorindex_vals, i)))));
      }
    //   std::cout << "returning second_lhs_tensor_multi_index_operand : "
    //             << second_lhs_tensor_multi_index_operand << std::endl;
      return second_lhs_tensor_multi_index_operand;
    }
  };

  //   get_first_tensor_index_operand(const std::array<size_t,
  //   num_tensor_indices>&
  //                                      lhs_tensor_multi_index) noexcept {
  //     // std::cout << "=== GET FIRST TENSOR INDEX OPERAND === " << std::endl;
  //     constexpr std::array<size_t, sizeof...(LhsIndices)>
  //     lhs_tensorindex_vals = {
  //         {LhsIndices::value...}};
  //     constexpr std::array<size_t, num_tensor_indices_first_operand>
  //         first_op_tensorindex_vals = {{Args1::value...}};
  //     std::array<size_t, num_tensor_indices_first_operand>
  //         first_tensor_index_operand;
  //     for (size_t i = 0; i < num_tensor_indices_first_operand; i++) {
  //       //   // next 4 lines will assign <pos of first op's index in LHS> of
  //       //   first_tensor_index_operand
  //       //   // to lhs_tensor_multi_index[first op's index]
  //       //   gsl::at(first_tensor_index_operand,
  //       //           // next 3 lines get position of first op's index in LHS
  //       //           static_cast<unsigned long>(std::distance(
  //       //               lhs_tensorindex_vals.begin(),
  //       //               alg::find(lhs_tensorindex_vals,
  //       //               gsl::at(first_op_tensorindex_vals, i))))) =
  //       //       gsl::at(lhs_tensor_multi_index, i);
  //       // next 4 lines will assign <pos of first op's index in LHS> of
  //       // first_tensor_index_operand to lhs_tensor_multi_index[first op's
  //       index] first_tensor_index_operand[i] =
  //           gsl::at(lhs_tensor_multi_index,
  //                   // next 3 lines get position of first op's index in LHS
  //                   static_cast<unsigned long>(std::distance(
  //                       lhs_tensorindex_vals.begin(),
  //                       alg::find(lhs_tensorindex_vals,
  //                                 gsl::at(first_op_tensorindex_vals, i)))));
  //     }
  //     return first_tensor_index_operand;
  //   }

  template <typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE static std::array<size_t,
                                          num_tensor_indices_second_operand>
  get_second_tensor_index_operand(const std::array<size_t, num_tensor_indices>&
                                      lhs_tensor_multi_index) noexcept {
    // std::cout << "=== GET SECOND TENSOR INDEX OPERAND === " << std::endl;
    constexpr std::array<size_t, sizeof...(LhsIndices)> lhs_tensorindex_vals = {
        {LhsIndices::value...}};
    // std::cout << "lhs_tensorindex_vals: " << lhs_tensorindex_vals <<
    // std::endl;
    constexpr std::array<size_t, num_tensor_indices_second_operand>
        second_op_tensorindex_vals = {{Args2::value...}};
    // std::cout << "second_op_tensorindex_vals: " << second_op_tensorindex_vals
    //           << std::endl;
    std::array<size_t, num_tensor_indices_second_operand>
        second_tensor_index_operand;
    for (size_t i = 0; i < num_tensor_indices_second_operand; i++) {
      //   std::cout << "i: " << i << std::endl;
      //   std::cout << "second_op_tensorindex_vals[i]: "
      //             << gsl::at(second_op_tensorindex_vals, i) << std::endl;
      //   std::cout << "index of second_op_tensor_index_vals[i]: "
      //             << static_cast<unsigned long>(std::distance(
      //                    lhs_tensorindex_vals.begin(),
      //                    alg::find(lhs_tensorindex_vals,
      //                              gsl::at(second_op_tensorindex_vals, i))))
      //             << std::endl;

      // next 4 lines will assign <pos of second op's index in LHS> of
      // second_tensor_index_operand to lhs_tensor_multi_index[second op's
      // index] gsl::at(second_tensor_index_operand,
      //         // next 3 lines get position of second op's index in LHS
      //         static_cast<unsigned long>(std::distance(
      //             lhs_tensorindex_vals.begin(),
      //             alg::find(lhs_tensorindex_vals,
      //             gsl::at(second_op_tensorindex_vals, i))))) =
      //     gsl::at(lhs_tensor_multi_index, i);

      // next 4 lines will assign <pos of second op's index in LHS> of
      // second_tensor_index_operand to lhs_tensor_multi_index[second op's
      // index]
      second_tensor_index_operand[i] =
          gsl::at(lhs_tensor_multi_index,
                  // next 3 lines get position of second op's index in LHS
                  static_cast<unsigned long>(std::distance(
                      lhs_tensorindex_vals.begin(),
                      alg::find(lhs_tensorindex_vals,
                                gsl::at(second_op_tensorindex_vals, i)))));
    }
    // std::cout << "=== RETURNING FROM GET SECOND TENSOR INDEX OPERAND === "
    //           << std::endl;
    return second_tensor_index_operand;
  }

  //   template <typename LhsIndexList>
  //   using get_first_op_tensorindex_list =
  //       tmpl::find<
  //           LhsIndexList,
  //           std::is_same<
  //               tmpl::index_of<
  //                   ArgsList2<Args2...>,
  //                   tmpl::_1
  //               >,
  //              tmpl::no_such_type_
  //           >
  //       >;
  //   template <typename LhsIndexList>
  //   using get_first_op_tensorindex_list =
  //       tmpl::find<LhsIndexList, std::is_same<tmpl::bind<tmpl::index_of,
  //       tmpl::pin<ArgsList2<Args2...>>, tmpl::_1>,
  //                                                    tmpl::no_such_type_>>;
  // template <typename LhsIndexList>
  // using get_first_op_tensorindex_list =
  //     tmpl::find<LhsIndexList, tmpl::bind<std::is_same,   tmpl::pin<
  //     tmpl::index_of<ArgsList1<Args1...>, tmpl::_1>  >, tmpl::no_such_type_
  //     >>;

  // template <typename LhsIndexList>
  // using get_first_op_tensorindex_list =
  //     tmpl::find<
  //         tmpl::bind<
  //             std::is_same,
  //             tmpl::bind<
  //                 tmpl::index_of, tmpl::pin<ArgsList2<Args2...>>, tmpl::_1
  //             >,
  //             tmpl::no_such_type_
  //         >
  //     >;

  template <typename LhsIndexList>
  using get_first_op_tensorindex_list = tmpl::filter<
      LhsIndexList,
      tmpl::bind<tmpl::found, tmpl::pin<ArgsList1<Args1...>>,
                 tmpl::bind<std::is_same, tmpl::_1, tmpl::parent<tmpl::_1>>>>;

  //   using mapping = tmpl::transform<
  //     lhs_indices, tmpl::bind<tmpl::index_of, tmpl::pin<rhs_indices>,
  //     tmpl::_1>>;

  //   template <typename LhsIndexList>
  //   using get_second_op_tensorindex_list =
  //       tmpl::find<LhsIndexList,
  //       std::is_same<tmpl::index_of<ArgsList1<Args1...>, tmpl::_1>,
  //                                                    tmpl::no_such_type_>>;

  template <typename LhsIndexList>
  using get_second_op_tensorindex_list = tmpl::filter<
      LhsIndexList,
      tmpl::bind<tmpl::found, tmpl::pin<ArgsList2<Args2...>>,
                 tmpl::bind<std::is_same, tmpl::_1, tmpl::parent<tmpl::_1>>>>;

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
        /*const std::array<size_t, num_tensor_indices_first_operand>&
            first_tensor_index_operand,
        const std::array<size_t, num_tensor_indices_second_operand>&
            second_tensor_index_operand,*/
        const size_t first_storage_index_operand,
        const size_t second_storage_index_operand, const T1& t1, const T2& t2) {
      //   auto res1 = (t1.template get<FirstOperandLhsTensorIndices...>(
      //              first_tensor_index_operand));
      //   auto res2 = (t2.template get<SecondOperandLhsTensorIndices...>(
      //              second_tensor_index_operand));
      //   std::cout << "first_tensor_index_operand : " <<
      //   first_tensor_index_operand << std::endl; std::cout << res1 <<
      //   std::endl; std::cout << "second_tensor_index_operand : " <<
      //   second_tensor_index_operand << std::endl; std::cout << res2 <<
      //   std::endl; std::cout << "product : " << res1 * res2 << std::endl;

      //   return t1.template get<FirstOperandLhsTensorIndices...>(
      //              first_tensor_index_operand) *
      //          t2.template get<SecondOperandLhsTensorIndices...>(
      //              second_tensor_index_operand);

    //   auto res1 = (t1.template get<UncontractedLhsStructureFirstOp,
    //                                FirstOperandLhsTensorIndices...>(
    //       first_storage_index_operand));
    //   auto res2 = (t2.template get<UncontractedLhsStructureSecondOp,
    //                                SecondOperandLhsTensorIndices...>(
    //       second_storage_index_operand));
    //   std::cout << "first_storage_index_operand : "
    //             << first_storage_index_operand << std::endl;
    //   std::cout << res1 << std::endl;
    //   std::cout << "second_storage_index_operand : "
    //             << second_storage_index_operand << std::endl;
    //   std::cout << res2 << std::endl;
    //   std::cout << "product : " << res1 * res2 << std::endl;

      return t1.template get<UncontractedLhsStructureFirstOp,
                             FirstOperandLhsTensorIndices...>(
                 first_storage_index_operand) *
             t2.template get<UncontractedLhsStructureSecondOp,
                             SecondOperandLhsTensorIndices...>(
                 second_storage_index_operand);
    }
  };

  // TODO: The args will need to be reduced in a careful manner, which means
  // they need to be reduced together, then split at the correct length so that
  // the indexing is correct.
  template <typename... LhsIndices, typename U>
  SPECTRE_ALWAYS_INLINE /*type*/ decltype(auto) get(
      const std::array<U, num_tensor_indices>& lhs_tensor_multi_index) const {
    // return t1_.template get<LhsIndices...>(tensor_index) *
    //        t2_.template get<LhsIndices...>(tensor_index);
    using first_op_tensorindex_list =
        get_first_op_tensorindex_list<tmpl::list<LhsIndices...>>;
    // td<first_op_tensorindex_list> first;
    using second_op_tensorindex_list =
        get_second_op_tensorindex_list<tmpl::list<LhsIndices...>>;
    // td<second_op_tensorindex_list> second;

    // std::cout << "=== PRODUCT GET === " << std::endl;
    std::array<size_t, num_tensor_indices_first_operand>
        first_tensor_index_operand =
            GetFirstTensorMultiIndexOperand<first_op_tensorindex_list>::
                template apply<LhsIndices...>(lhs_tensor_multi_index);
    // get_first_tensor_index_operand<LhsIndices...>(
    //     lhs_tensor_multi_index);
    std::array<size_t, num_tensor_indices_second_operand>
        second_tensor_index_operand =
            GetSecondTensorMultiIndexOperand<second_op_tensorindex_list>::
                template apply<LhsIndices...>(lhs_tensor_multi_index);
    // get_second_tensor_index_operand<LhsIndices...>(
    //     lhs_tensor_multi_index);

    // std::cout << "lhs_tensor_multi_index: " << lhs_tensor_multi_index
    //           << std::endl;
    // std::cout << "first_tensor_index_operand: " << first_tensor_index_operand
    //           << std::endl;
    // std::cout << "second_tensor_index_operand: "
    //           << second_tensor_index_operand
    //           << std::endl
    //           << std::endl;
    // return t1_.template get<LhsIndices...>(first_tensor_index_operand) *
    //        t2_.template get<LhsIndices...>(second_tensor_index_operand);

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
    // return ComputeProduct<
    //     first_op_tensorindex_list,
    //     second_op_tensorindex_list>::apply(first_tensor_index_operand,
    //                                        second_tensor_index_operand, t1_,
    //                                        t2_);
    return ComputeProduct<first_op_tensorindex_list,
                          second_op_tensorindex_list>::
        template apply<uncontracted_lhs_structure_first_op,
                       uncontracted_lhs_structure_second_op>(
            first_storage_index_operand, second_storage_index_operand, t1_,
            t2_);
  }

  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const size_t lhs_storage_index) const {
    const std::array<size_t, num_tensor_indices>& lhs_tensor_multi_index =
        LhsStructure::get_canonical_tensor_index(lhs_storage_index);

    // return t1_.template get<LhsIndices...>(tensor_index) *
    //        t2_.template get<LhsIndices...>(tensor_index);
    using first_op_tensorindex_list =
        get_first_op_tensorindex_list<tmpl::list<LhsIndices...>>;
    // td<first_op_tensorindex_list> first;
    using second_op_tensorindex_list =
        get_second_op_tensorindex_list<tmpl::list<LhsIndices...>>;
    // td<second_op_tensorindex_list> second;

    // std::cout << "=== PRODUCT GET === " << std::endl;
    std::array<size_t, num_tensor_indices_first_operand>
        first_tensor_index_operand =
            GetFirstTensorMultiIndexOperand<first_op_tensorindex_list>::
                template apply<LhsIndices...>(lhs_tensor_multi_index);
    // get_first_tensor_index_operand<LhsIndices...>(
    //     lhs_tensor_multi_index);
    std::array<size_t, num_tensor_indices_second_operand>
        second_tensor_index_operand =
            GetSecondTensorMultiIndexOperand<second_op_tensorindex_list>::
                template apply<LhsIndices...>(lhs_tensor_multi_index);
    // get_second_tensor_index_operand<LhsIndices...>(
    //     lhs_tensor_multi_index);

    // std::cout << "lhs_tensor_multi_index: " << lhs_tensor_multi_index
    //           << std::endl;
    // std::cout << "first_tensor_index_operand: " << first_tensor_index_operand
    //           << std::endl;
    // std::cout << "second_tensor_index_operand: "
    //           << second_tensor_index_operand
    //           << std::endl
    //           << std::endl;
    // return t1_.template get<LhsIndices...>(first_tensor_index_operand) *
    //        t2_.template get<LhsIndices...>(second_tensor_index_operand);

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
    // return ComputeProduct<
    //     first_op_tensorindex_list,
    //     second_op_tensorindex_list>::apply(first_tensor_index_operand,
    //                                        second_tensor_index_operand, t1_,
    //                                        t2_);
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

template <class... T>
struct td;

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
  // static_assert(tmpl::size<Args1>::value == tmpl::size<Args2>::value,
  //               "Tensor addition is only possible with the same rank
  //               tensors");
  // static_assert(tmpl::equal_members<Args1, Args2>::value,
  //               "The indices when adding two tensors must be equal. This
  //               error "
  //               "occurs from expressions like A(_a, _b) + B(_c, _a)");

  //   return TensorExpressions::Product<
  //       typename std::conditional<
  //           std::is_base_of<Expression, T1>::value, T1,
  //           TensorExpression<T1, X, Symm1, IndexList1, Args1>>::type,
  //       typename std::conditional<
  //           std::is_base_of<Expression, T2>::value, T2,
  //           TensorExpression<T2, X, Symm2, IndexList2, Args2>>::type,
  //       Args1, Args2>(~t1, ~t2);

  auto prod_expr = TensorExpressions::Product<
      typename std::conditional<
          std::is_base_of<Expression, T1>::value, T1,
          TensorExpression<T1, X, Symm1, IndexList1, Args1>>::type,
      typename std::conditional<
          std::is_base_of<Expression, T2>::value, T2,
          TensorExpression<T2, X, Symm2, IndexList2, Args2>>::type,
      Args1, Args2>(~t1, ~t2);

  // td<decltype(prod_expr)> idk;
  // /home/alexmacedo/spectre/src/DataStructures/Tensor/Expressions/
  // Product.hpp:447:29:
  // error: implicit instantiation of undefined template
  // 'td<TensorExpressions::Product<TensorExpression<Tensor<double,
  // brigand::list<brigand::integral_constant<int, 1> >,
  // brigand::list<Tensor_detail::TensorIndexType<3, UpLo::Up, Frame::Grid,
  // IndexType::Spacetime> > >, double,
  // brigand::list<brigand::integral_constant<int, 1> >,
  // brigand::list<Tensor_detail::TensorIndexType<3, UpLo::Up, Frame::Grid,
  // IndexType::Spacetime> >, brigand::list<TensorIndex<500, nullptr> >,
  // brigand::list<> >, TensorExpression<Tensor<double,
  // brigand::list<brigand::integral_constant<int, 1> >,
  // brigand::list<Tensor_detail::TensorIndexType<3, UpLo::Up, Frame::Grid,
  // IndexType::Spacetime> > >, double,
  // brigand::list<brigand::integral_constant<int, 1> >,
  // brigand::list<Tensor_detail::TensorIndexType<3, UpLo::Up, Frame::Grid,
  // IndexType::Spacetime> >, brigand::list<TensorIndex<501, nullptr> >,
  // brigand::list<> >, brigand::list<TensorIndex<500, nullptr> >,
  // brigand::list<TensorIndex<501, nullptr> > > >' td<decltype(prod_expr)> idk;
  return TensorExpressions::contract(prod_expr);

  //   return TensorExpressions::contract(
  //       TensorExpression<Tensor<X, Symm, tmpl::list<Indices...>>, X, Symm,
  //                               tmpl::list<Indices...>, ArgsList>{
  //           <product expression here>
  //       }
  //   );

  // return
  // TensorExpressions::contract(TE<tmpl::list<TensorIndices...>>{*this});
  //   template <typename ArgsList>
  //   using TE = TensorExpression<Tensor<X, Symm, tmpl::list<Indices...>>, X,
  //   Symm,
  //                               tmpl::list<Indices...>, ArgsList>;

  // return contract(TensorExpressions::Product<
  //     typename std::conditional<
  //         std::is_base_of<Expression, T1>::value, T1,
  //         TensorExpression<T1, X, Symm1, IndexList1, Args1>>::type,
  //     typename std::conditional<
  //         std::is_base_of<Expression, T2>::value, T2,
  //         TensorExpression<T2, X, Symm2, IndexList2, Args2>>::type,
  //    Args1, Args2>(~t1, ~t2));
}

// From discussion with Nils:
// auto operator*(const TensorExp1& lhs, const TensorExp2& rhs) noexcept {
//   return contract(Product<...>{~lhs, ~rhs});
// }
