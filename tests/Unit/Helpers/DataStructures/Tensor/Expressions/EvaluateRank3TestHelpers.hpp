// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 3 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a_t`, `ti_b_t`, and `ti_c_t`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used.
///
/// If we consider the RHS tensor's generic indices to be (a, b, c), then this
/// test checks that the data in the evaluated LHS tensor is correct according
/// to the index orders of the LHS and RHS. The possible cases that are checked
/// are when the LHS tensor is evaluated with index orders: (a, b, c),
/// (a, c, b), (b, a, c), (b, c, a), (c, a, b), and (c, b, a).
///
/// \tparam Datatype the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam TensorIndexA the type of the first TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_a_t`
/// \tparam TensorIndexB the type of the second TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_B_t`
/// \tparam TensorIndexC the type of the third TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_c_t`
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
template <typename Datatype, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_evaluate_rank_3_core(const TensorIndexA& tensorindex_a,
                               const TensorIndexB& tensorindex_b,
                               const TensorIndexC& tensorindex_c) {
  Tensor<Datatype, RhsSymmetry, RhsTensorIndexTypeList> rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);

  size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;
  size_t dim_c = tmpl::at_c<RhsTensorIndexTypeList, 2>::dim;

  // L_{abc} = R_{abc}
  auto abc_to_abc =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  // L_{acb} = R_{abc}
  auto abc_to_acb =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  // L_{bac} = R_{abc}
  auto abc_to_bac =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  // L_{bca} = R_{abc}
  auto abc_to_bca =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  // L_{cab} = R_{abc}
  auto abc_to_cab =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  // L_{cba} = R_{abc}
  auto abc_to_cba =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c));

  for (size_t i = 0; i < dim_a; ++i) {
    for (size_t j = 0; j < dim_b; ++j) {
      for (size_t k = 0; k < dim_c; ++k) {
        CHECK(abc_to_abc.get(i, j, k) == rhs_tensor.get(i, j, k));
        CHECK(abc_to_acb.get(i, k, j) == rhs_tensor.get(i, j, k));
        CHECK(abc_to_bac.get(j, i, k) == rhs_tensor.get(i, j, k));
        CHECK(abc_to_bca.get(j, k, i) == rhs_tensor.get(i, j, k));
        CHECK(abc_to_cab.get(k, i, j) == rhs_tensor.get(i, j, k));
        CHECK(abc_to_cba.get(k, j, i) == rhs_tensor.get(i, j, k));
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluate for single rank 3 Tensors on multiple
/// Frame types and dimension combinations for nonsymmetric indices
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a_t`, `ti_b_t`, and `ti_c_t`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used..
///
/// \tparam Datatype the type of data being stored in the Tensors
/// \tparam TensorIndexTypeA the \ref SpacetimeIndex "TensorIndexType" of the
/// first index of the RHS Tensor
/// \tparam TensorIndexTypeB the \ref SpacetimeIndex "TensorIndexType" of the
/// second index of the RHS Tensor
/// \tparam TensorIndexTypeC the \ref SpacetimeIndex "TensorIndexType" of the
/// third index of the RHS Tensor
/// \tparam TensorIndexA the type of the first TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_a_t`
/// \tparam TensorIndexB the type of the second TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_B_t`
/// \tparam TensorIndexC the type of the third TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_c_t`
/// \tparam ValenceA the valence of the first index used on the RHS of the
/// TensorExpression
/// \tparam ValenceB the valence of the second index used on the RHS of the
/// TensorExpression
/// \tparam ValenceC the valence of the third index used on the RHS of the
/// TensorExpression
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
template <typename Datatype,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC,
          typename TensorIndexA, typename TensorIndexB, typename TensorIndexC>
void test_evaluate_rank_3_no_symmetry(const TensorIndexA& tensorindex_a,
                                      const TensorIndexB& tensorindex_b,
                                      const TensorIndexC& tensorindex_c) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(2, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(3, data)

#define CALL_TEST_EVALUATE_RANK_3_CORE(_, data)                            \
  test_evaluate_rank_3_core<Datatype, Symmetry<3, 2, 1>,                   \
                            index_list<                                    \
                                TensorIndexTypeA<                          \
                                    DIM_A(data), ValenceA, FRAME(data)>,   \
                                TensorIndexTypeB<                          \
                                    DIM_B(data), ValenceB, FRAME(data)>,   \
                                TensorIndexTypeC<                          \
                                    DIM_C(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_CORE, (1, 2, 3),
                          (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_CORE
#undef FRAME
#undef DIM_C
#undef DIM_B
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluate for single rank 3 Tensors on multiple
/// Frame types and dimension combinations for right hand side tensors whose
/// first and second indices are symmetric
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a_t`, `ti_b_t`, and `ti_c_t`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used..
///
/// \tparam Datatype the type of data being stored in the Tensors
/// \tparam TensorIndexTypeA the \ref SpacetimeIndex "TensorIndexType" of the
/// first index of the RHS Tensor
/// \tparam TensorIndexTypeB the \ref SpacetimeIndex "TensorIndexType" of the
/// second index of the RHS Tensor
/// \tparam TensorIndexTypeC the \ref SpacetimeIndex "TensorIndexType" of the
/// third index of the RHS Tensor
/// \tparam TensorIndexA the type of the first TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_a_t`
/// \tparam TensorIndexB the type of the second TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_B_t`
/// \tparam TensorIndexC the type of the third TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_c_t`
/// \tparam ValenceA the valence of the first index used on the RHS of the
/// TensorExpression
/// \tparam ValenceB the valence of the second index used on the RHS of the
/// TensorExpression
/// \tparam ValenceC the valence of the third index used on the RHS of the
/// TensorExpression
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
template <typename Datatype,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC,
          typename TensorIndexA, typename TensorIndexB, typename TensorIndexC>
void test_evaluate_rank_3_ab_symmetry(const TensorIndexA& tensorindex_a,
                                      const TensorIndexB& tensorindex_b,
                                      const TensorIndexC& tensorindex_c) {
#define DIM_AB(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_CORE(_, data)                            \
  test_evaluate_rank_3_core<Datatype, Symmetry<2, 2, 1>,                   \
                            index_list<                                    \
                                TensorIndexTypeA<                          \
                                    DIM_AB(data), ValenceA, FRAME(data)>,  \
                                TensorIndexTypeB<                          \
                                    DIM_AB(data), ValenceB, FRAME(data)>,  \
                                TensorIndexTypeC<                          \
                                    DIM_C(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_CORE, (1, 2, 3),
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_CORE
#undef FRAME
#undef DIM_C
#undef DIM_AB
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluate for single rank 3 Tensors on multiple
/// Frame types and dimension combinations for right hand side tensors whose
/// first and third indices are symmetric
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a_t`, `ti_b_t`, and `ti_c_t`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used..
///
/// \tparam Datatype the type of data being stored in the Tensors
/// \tparam TensorIndexTypeA the \ref SpacetimeIndex "TensorIndexType" of the
/// first index of the RHS Tensor
/// \tparam TensorIndexTypeB the \ref SpacetimeIndex "TensorIndexType" of the
/// second index of the RHS Tensor
/// \tparam TensorIndexTypeC the \ref SpacetimeIndex "TensorIndexType" of the
/// third index of the RHS Tensor
/// \tparam TensorIndexA the type of the first TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_a_t`
/// \tparam TensorIndexB the type of the second TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_B_t`
/// \tparam TensorIndexC the type of the third TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_c_t`
/// \tparam ValenceA the valence of the first index used on the RHS of the
/// TensorExpression
/// \tparam ValenceB the valence of the second index used on the RHS of the
/// TensorExpression
/// \tparam ValenceC the valence of the third index used on the RHS of the
/// TensorExpression
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
template <typename Datatype,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC,
          typename TensorIndexA, typename TensorIndexB, typename TensorIndexC>
void test_evaluate_rank_3_ac_symmetry(const TensorIndexA& tensorindex_a,
                                      const TensorIndexB& tensorindex_b,
                                      const TensorIndexC& tensorindex_c) {
#define DIM_AC(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_CORE(_, data)                             \
  test_evaluate_rank_3_core<Datatype, Symmetry<2, 1, 2>,                    \
                            index_list<                                     \
                                TensorIndexTypeA<                           \
                                    DIM_AC(data), ValenceA, FRAME(data)>,   \
                                TensorIndexTypeB<                           \
                                    DIM_B(data), ValenceB, FRAME(data)>,    \
                                TensorIndexTypeC<                           \
                                    DIM_AC(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_CORE, (1, 2, 3),
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_CORE
#undef FRAME
#undef DIM_B
#undef DIM_AC
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluate for single rank 3 Tensors on multiple
/// Frame types and dimension combinations for right hand side tensors whose
/// second and third indices are symmetric
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a_t`, `ti_b_t`, and `ti_c_t`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used..
///
/// \tparam Datatype the type of data being stored in the Tensors
/// \tparam TensorIndexTypeA the \ref SpacetimeIndex "TensorIndexType" of the
/// first index of the RHS Tensor
/// \tparam TensorIndexTypeB the \ref SpacetimeIndex "TensorIndexType" of the
/// second index of the RHS Tensor
/// \tparam TensorIndexTypeC the \ref SpacetimeIndex "TensorIndexType" of the
/// third index of the RHS Tensor
/// \tparam TensorIndexA the type of the first TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_a_t`
/// \tparam TensorIndexB the type of the second TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_B_t`
/// \tparam TensorIndexC the type of the third TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_c_t`
/// \tparam ValenceA the valence of the first index used on the RHS of the
/// TensorExpression
/// \tparam ValenceB the valence of the second index used on the RHS of the
/// TensorExpression
/// \tparam ValenceC the valence of the third index used on the RHS of the
/// TensorExpression
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
template <typename Datatype,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC,
          typename TensorIndexA, typename TensorIndexB, typename TensorIndexC>
void test_evaluate_rank_3_bc_symmetry(const TensorIndexA& tensorindex_a,
                                      const TensorIndexB& tensorindex_b,
                                      const TensorIndexC& tensorindex_c) {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_BC(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_CORE(_, data)                             \
  test_evaluate_rank_3_core<Datatype, Symmetry<2, 1, 1>,                    \
                            index_list<                                     \
                                TensorIndexTypeA<                           \
                                    DIM_A(data), ValenceA, FRAME(data)>,    \
                                TensorIndexTypeB<                           \
                                    DIM_BC(data), ValenceB, FRAME(data)>,   \
                                TensorIndexTypeC<                           \
                                    DIM_BC(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_CORE, (1, 2, 3),
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_CORE
#undef FRAME
#undef DIM_BC
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluate for single rank 3 Tensors on multiple
/// Frame types and dimension combinations for symmetric indices
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a_t`, `ti_b_t`, and `ti_c_t`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used..
///
/// \tparam Datatype the type of data being stored in the Tensors
/// \tparam TensorIndexTypeA the \ref SpacetimeIndex "TensorIndexType" of the
/// first index of the RHS Tensor
/// \tparam TensorIndexTypeB the \ref SpacetimeIndex "TensorIndexType" of the
/// second index of the RHS Tensor
/// \tparam TensorIndexTypeC the \ref SpacetimeIndex "TensorIndexType" of the
/// third index of the RHS Tensor
/// \tparam TensorIndexA the type of the first TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_a_t`
/// \tparam TensorIndexB the type of the second TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_B_t`
/// \tparam TensorIndexC the type of the third TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_c_t`
/// \tparam ValenceA the valence of the first index used on the RHS of the
/// TensorExpression
/// \tparam ValenceB the valence of the second index used on the RHS of the
/// TensorExpression
/// \tparam ValenceC the valence of the third index used on the RHS of the
/// TensorExpression
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
template <typename Datatype,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC,
          typename TensorIndexA, typename TensorIndexB, typename TensorIndexC>
void test_evaluate_rank_3_abc_symmetry(const TensorIndexA& tensorindex_a,
                                       const TensorIndexB& tensorindex_b,
                                       const TensorIndexC& tensorindex_c) {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_CORE(_, data)                          \
  test_evaluate_rank_3_core<Datatype, Symmetry<1, 1, 1>,                 \
                            index_list<                                  \
                                TensorIndexTypeA<                        \
                                    DIM(data), ValenceA, FRAME(data)>,   \
                                TensorIndexTypeB<                        \
                                    DIM(data), ValenceB, FRAME(data)>,   \
                                TensorIndexTypeC<                        \
                                    DIM(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_CORE, (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_CORE
#undef FRAME
#undef DIM
}
