// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename... Ts>
void create_tensor(gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void create_tensor(gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the outer product of a rank 0 tensor with another tensor is
/// correctly evaluated
///
/// \details
/// The outer product cases tested are:
/// - (rank 0) = (rank 0) x (rank 0)
/// - (rank 0) = (rank 0) x (rank 0) x (rank 0)
/// - (rank 1) = (rank 0) x (rank 1)
/// - (rank 1) = (rank 1) x (rank 0)
/// - (rank 2) = (rank 0) x (rank 2)
/// - (rank 2) = (rank 2) x (rank 0)
///
/// For the last two cases, both LHS index orderings are tested.
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_rank_0_outer_product(const DataType& used_for_size) noexcept {
  Tensor<DataType> R{{{used_for_size}}};
  if constexpr (std::is_same_v<DataType, double>) {
    // Instead of the tensor's value being the whole number, `used_for_size`
    R.get() = -3.7;
  } else {
    // Instead of the tensor's `DataVector` having elements with all the same
    // whole number value
    create_tensor(make_not_null(&R));
  }

  // \f$L = R * R\f$
  CHECK(TensorExpressions::evaluate(R() * R()).get() == R.get() * R.get());
  // \f$L = R * R * R\f$
  CHECK(TensorExpressions::evaluate(R() * R() * R()).get() ==
        R.get() * R.get() * R.get());

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Su(used_for_size);
  create_tensor(make_not_null(&Su));

  // \f$L^{a} = R * S^{a}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const decltype(Su) LA_from_R_SA =
      TensorExpressions::evaluate<ti_A>(R() * Su(ti_A));
  // \f$L^{a} = S^{a} * R\f$
  const decltype(Su) LA_from_SA_R =
      TensorExpressions::evaluate<ti_A>(Su(ti_A) * R());

  for (size_t a = 0; a < 4; a++) {
    CHECK(LA_from_R_SA.get(a) == R.get() * Su.get(a));
    CHECK(LA_from_SA_R.get(a) == Su.get(a) * R.get());
  }

  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      Tll(used_for_size);
  create_tensor(make_not_null(&Tll));

  // \f$L_{ai} = R * T_{ai}\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_Tai =
          TensorExpressions::evaluate<ti_a, ti_i>(R() * Tll(ti_a, ti_i));
  // \f$L_{ia} = R * T_{ai}\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_Tai =
          TensorExpressions::evaluate<ti_i, ti_a>(R() * Tll(ti_a, ti_i));
  // \f$L_{ai} = T_{ai} * R\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      Lai_from_Tai_R =
          TensorExpressions::evaluate<ti_a, ti_i>(Tll(ti_a, ti_i) * R());
  // \f$L_{ia} = T_{ai} * R\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      Lia_from_Tai_R =
          TensorExpressions::evaluate<ti_i, ti_a>(Tll(ti_a, ti_i) * R());

  for (size_t a = 0; a < 4; a++) {
    for (size_t i = 0; i < 4; i++) {
      const DataType expected_R_Tai_product = R.get() * Tll.get(a, i);
      CHECK(Lai_from_R_Tai.get(a, i) == expected_R_Tai_product);
      CHECK(Lia_from_R_Tai.get(i, a) == expected_R_Tai_product);

      const DataType expected_Tai_R_product = Tll.get(a, i) * R.get();
      CHECK(Lai_from_Tai_R.get(a, i) == expected_Tai_R_product);
      CHECK(Lia_from_Tai_R.get(i, a) == expected_Tai_R_product);
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the outer product of rank 1 tensors with another tensor is
/// correctly evaluated
///
/// \details
/// The outer product cases tested are:
/// - (rank 2) = (rank 1) x (rank 1)
/// - (rank 3) = (rank 1) x (rank 1) x (rank 1)
/// - (rank 3) = (rank 1) x (rank 2)
/// - (rank 3) = (rank 2) x (rank 1)
///
/// For each case, the LHS index order is different from the order in the
/// operands.
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_rank_1_outer_product(const DataType& used_for_size) noexcept {
  Tensor<DataType, Symmetry<1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Rl(used_for_size);
  create_tensor(make_not_null(&Rl));

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Su(used_for_size);
  create_tensor(make_not_null(&Su));

  Tensor<DataType, Symmetry<1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      Tu(used_for_size);
  create_tensor(make_not_null(&Tu));

  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Gll(used_for_size);
  create_tensor(make_not_null(&Gll));

  // \f$L^{a}{}_{i} = R_{i} * S^{a}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      LAi_from_Ri_SA =
          TensorExpressions::evaluate<ti_A, ti_i>(Rl(ti_i) * Su(ti_A));

  for (size_t i = 0; i < 3; i++) {
    for (size_t a = 0; a < 4; a++) {
      CHECK(LAi_from_Ri_SA.get(a, i) == Rl.get(i) * Su.get(a));
    }
  }

  // \f$L_{ja}{}^{i} = R_{i} * S^{a} * T^{j}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      LJAi_from_Ri_SA_TJ = TensorExpressions::evaluate<ti_J, ti_A, ti_i>(
          Rl(ti_i) * Su(ti_A) * Tu(ti_J));

  for (size_t j = 0; j < 3; j++) {
    for (size_t a = 0; a < 4; a++) {
      for (size_t i = 0; i < 3; i++) {
        CHECK(LJAi_from_Ri_SA_TJ.get(j, a, i) ==
              Rl.get(i) * Su.get(a) * Tu.get(j));
      }
    }
  }

  // \f$L_{k}{}^{c}{}_{d} = S^{c} * G_{dk}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      LkCd_from_SC_Gdk = TensorExpressions::evaluate<ti_k, ti_C, ti_d>(
          Su(ti_C) * Gll(ti_d, ti_k));
  // \f$L^{c}{}_{dk} = G_{dk} * S^{c}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      LCdk_from_Gdk_SC = TensorExpressions::evaluate<ti_C, ti_d, ti_k>(
          Gll(ti_d, ti_k) * Su(ti_C));

  for (size_t k = 0; k < 4; k++) {
    for (size_t c = 0; c < 4; c++) {
      for (size_t d = 0; d < 4; d++) {
        CHECK(LkCd_from_SC_Gdk.get(k, c, d) == Su.get(c) * Gll.get(d, k));
        CHECK(LCdk_from_Gdk_SC.get(c, d, k) == Gll.get(d, k) * Su.get(c));
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the outer product of two rank 2 tensors is correctly evaluated
///
/// \details
/// All LHS index orders are tested.
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_rank_2_outer_product(const DataType& used_for_size) noexcept {
  using R_index = SpacetimeIndex<3, UpLo::Lo, Frame::Grid>;
  using S_first_index = SpatialIndex<4, UpLo::Up, Frame::Grid>;
  using S_second_index = SpacetimeIndex<2, UpLo::Lo, Frame::Grid>;

  Tensor<DataType, Symmetry<1, 1>, index_list<R_index, R_index>> Rll(
      used_for_size);
  create_tensor(make_not_null(&Rll));
  Tensor<DataType, Symmetry<2, 1>, index_list<S_first_index, S_second_index>>
      Sul(used_for_size);
  create_tensor(make_not_null(&Sul));

  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<3, 3, 2, 1>,
               index_list<R_index, R_index, S_first_index, S_second_index>>
      L_abIc = TensorExpressions::evaluate<ti_a, ti_b, ti_I, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 3, 2, 1>,
               index_list<R_index, R_index, S_second_index, S_first_index>>
      L_abcI = TensorExpressions::evaluate<ti_a, ti_b, ti_c, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 3, 1>,
               index_list<R_index, S_first_index, R_index, S_second_index>>
      L_aIbc = TensorExpressions::evaluate<ti_a, ti_I, ti_b, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 3>,
               index_list<R_index, S_first_index, S_second_index, R_index>>
      L_aIcb = TensorExpressions::evaluate<ti_a, ti_I, ti_c, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 3, 1>,
               index_list<R_index, S_second_index, R_index, S_first_index>>
      L_acbI = TensorExpressions::evaluate<ti_a, ti_c, ti_b, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 3>,
               index_list<R_index, S_second_index, S_first_index, R_index>>
      L_acIb = TensorExpressions::evaluate<ti_a, ti_c, ti_I, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));

  const Tensor<DataType, Symmetry<3, 3, 2, 1>,
               index_list<R_index, R_index, S_first_index, S_second_index>>
      L_baIc = TensorExpressions::evaluate<ti_b, ti_a, ti_I, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 3, 2, 1>,
               index_list<R_index, R_index, S_second_index, S_first_index>>
      L_bacI = TensorExpressions::evaluate<ti_b, ti_a, ti_c, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 3, 1>,
               index_list<R_index, S_first_index, R_index, S_second_index>>
      L_bIac = TensorExpressions::evaluate<ti_b, ti_I, ti_a, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 3>,
               index_list<R_index, S_first_index, S_second_index, R_index>>
      L_bIca = TensorExpressions::evaluate<ti_b, ti_I, ti_c, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 3, 1>,
               index_list<R_index, S_second_index, R_index, S_first_index>>
      L_bcaI = TensorExpressions::evaluate<ti_b, ti_c, ti_a, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 3>,
               index_list<R_index, S_second_index, S_first_index, R_index>>
      L_bcIa = TensorExpressions::evaluate<ti_b, ti_c, ti_I, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));

  const Tensor<DataType, Symmetry<3, 2, 2, 1>,
               index_list<S_first_index, R_index, R_index, S_second_index>>
      L_Iabc = TensorExpressions::evaluate<ti_I, ti_a, ti_b, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 2>,
               index_list<S_first_index, R_index, S_second_index, R_index>>
      L_Iacb = TensorExpressions::evaluate<ti_I, ti_a, ti_c, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 2, 1>,
               index_list<S_first_index, R_index, R_index, S_second_index>>
      L_Ibac = TensorExpressions::evaluate<ti_I, ti_b, ti_a, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 2>,
               index_list<S_first_index, R_index, S_second_index, R_index>>
      L_Ibca = TensorExpressions::evaluate<ti_I, ti_b, ti_c, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 1>,
               index_list<S_first_index, S_second_index, R_index, R_index>>
      L_Icab = TensorExpressions::evaluate<ti_I, ti_c, ti_a, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 1>,
               index_list<S_first_index, S_second_index, R_index, R_index>>
      L_Icba = TensorExpressions::evaluate<ti_I, ti_c, ti_b, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));

  const Tensor<DataType, Symmetry<3, 2, 2, 1>,
               index_list<S_second_index, R_index, R_index, S_first_index>>
      L_cabI = TensorExpressions::evaluate<ti_c, ti_a, ti_b, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 2>,
               index_list<S_second_index, R_index, S_first_index, R_index>>
      L_caIb = TensorExpressions::evaluate<ti_c, ti_a, ti_I, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 2, 1>,
               index_list<S_second_index, R_index, R_index, S_first_index>>
      L_cbaI = TensorExpressions::evaluate<ti_c, ti_b, ti_a, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 2>,
               index_list<S_second_index, R_index, S_first_index, R_index>>
      L_cbIa = TensorExpressions::evaluate<ti_c, ti_b, ti_I, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 1>,
               index_list<S_second_index, S_first_index, R_index, R_index>>
      L_cIab = TensorExpressions::evaluate<ti_c, ti_I, ti_a, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 1>,
               index_list<S_second_index, S_first_index, R_index, R_index>>
      L_cIba = TensorExpressions::evaluate<ti_c, ti_I, ti_b, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));

  for (size_t a = 0; a < R_index::dim; a++) {
    for (size_t b = 0; b < R_index::dim; b++) {
      for (size_t i = 0; i < S_first_index::dim; i++) {
        for (size_t c = 0; c < S_second_index::dim; c++) {
          CHECK(L_abIc.get(a, b, i, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_abcI.get(a, b, c, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_aIbc.get(a, i, b, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_aIcb.get(a, i, c, b) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_acbI.get(a, c, b, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_acIb.get(a, c, i, b) == Rll.get(a, b) * Sul.get(i, c));

          CHECK(L_baIc.get(b, a, i, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_bacI.get(b, a, c, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_bIac.get(b, i, a, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_bIca.get(b, i, c, a) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_bcaI.get(b, c, a, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_bcIa.get(b, c, i, a) == Rll.get(a, b) * Sul.get(i, c));

          CHECK(L_Iabc.get(i, a, b, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_Iacb.get(i, a, c, b) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_Ibac.get(i, b, a, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_Ibca.get(i, b, c, a) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_Icab.get(i, c, a, b) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_Icba.get(i, c, b, a) == Rll.get(a, b) * Sul.get(i, c));

          CHECK(L_cabI.get(c, a, b, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_caIb.get(c, a, i, b) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_cbaI.get(c, b, a, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_cbIa.get(c, b, i, a) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_cIab.get(c, i, a, b) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_cIba.get(c, i, b, a) == Rll.get(a, b) * Sul.get(i, c));
        }
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the outer product of a rank 0, rank 1, and rank 2 tensor is
/// correctly evaluated
///
/// \details
/// The outer product cases tested are:
/// - (rank 3) = (rank 0) x (rank 1) x (rank 2)
/// - (rank 3) = (rank 0) x (rank 2) x (rank 1)
/// - (rank 3) = (rank 1) x (rank 0) x (rank 2)
/// - (rank 3) = (rank 1) x (rank 2) x (rank 0)
/// - (rank 3) = (rank 2) x (rank 0) x (rank 1)
/// - (rank 3) = (rank 2) x (rank 1) x (rank 0)
///
/// For all cases, all LHS index orderings are tested.
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_ranks_0_1_2_outer_product(const DataType& used_for_size) noexcept {
  Tensor<DataType> R{{{used_for_size}}};
  if constexpr (std::is_same_v<DataType, double>) {
    // Instead of the tensor's value being the whole number, `used_for_size`
    R.get() = 4.3;
  } else {
    // Instead of the tensor's `DataVector` having elements with all the same
    // whole number value
    create_tensor(make_not_null(&R));
  }

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Su(used_for_size);
  create_tensor(make_not_null(&Su));

  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      Tll(used_for_size);
  create_tensor(make_not_null(&Tll));

  // \f$R * S^{a} * T_{bi}\f$
  const auto R_SA_Tbi_expr = R() * Su(ti_A) * Tll(ti_b, ti_i);
  // \f$L^{a}{}_{bi} = R * S^{a} * T_{bi}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LAbi_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(R_SA_Tbi_expr);
  // \f$L^{a}{}_{ib} = R * S^{a} * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LAib_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(R_SA_Tbi_expr);
  // \f$L_{b}{}^{a}{}_{i} = R * S^{a} * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LbAi_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(R_SA_Tbi_expr);
  // \f$L_{bi}{}^{a} = R * S^{a} * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LbiA_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(R_SA_Tbi_expr);
  // \f$L_{i}{}^{a}{}_{b} = R * S^{a} * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LiAb_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(R_SA_Tbi_expr);
  // \f$L_{ib}{}^{a} = R * S^{a} * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LibA_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(R_SA_Tbi_expr);

  // \f$R * T_{bi} * S^{a}\f$
  const auto R_Tbi_SA_expr = R() * Tll(ti_b, ti_i) * Su(ti_A);
  // \f$L^{a}{}_{bi} = R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LAbi_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(R_Tbi_SA_expr);
  // \f$L^{a}{}_{ib} = R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LAib_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(R_Tbi_SA_expr);
  // \f$L_{b}{}^{a}{}_{i} = R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LbAi_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(R_Tbi_SA_expr);
  // \f$L_{bi}{}^{a} = R * R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LbiA_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(R_Tbi_SA_expr);
  // \f$L_{i}{}^{a}{}_{b} = R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LiAb_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(R_Tbi_SA_expr);
  // \f$L_{ib}{}^{a} = R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LibA_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(R_Tbi_SA_expr);

  // \f$S^{a} * R * T_{bi}\f$
  const auto SA_R_Tbi_expr = Su(ti_A) * R() * Tll(ti_b, ti_i);
  // \f$L^{a}{}_{bi} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LAbi_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(SA_R_Tbi_expr);
  // \f$L^{a}{}_{ib} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LAib_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(SA_R_Tbi_expr);
  // \f$L_{b}{}^{a}{}_{i} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LbAi_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(SA_R_Tbi_expr);
  // \f$L_{bi}{}^{a} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LbiA_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(SA_R_Tbi_expr);
  // \f$L_{i}{}^{a}{}_{b} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LiAb_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(SA_R_Tbi_expr);
  // \f$L_{ib}{}^{a} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LibA_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(SA_R_Tbi_expr);

  // \f$S^{a} * T_{bi} * R\f$
  const auto SA_Tbi_R_expr = Su(ti_A) * Tll(ti_b, ti_i) * R();
  // \f$L^{a}{}_{bi} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LAbi_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(SA_Tbi_R_expr);
  // \f$L^{a}{}_{ib} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LAib_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(SA_Tbi_R_expr);
  // \f$L_{b}{}^{a}{}_{i} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LbAi_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(SA_Tbi_R_expr);
  // \f$L_{bi}{}^{a} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LbiA_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(SA_Tbi_R_expr);
  // \f$L_{i}{}^{a}{}_{b} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LiAb_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(SA_Tbi_R_expr);
  // \f$L_{ib}{}^{a} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LibA_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(SA_Tbi_R_expr);

  // \f$T_{bi} * R * S^{a}\f$
  const auto Tbi_R_SA_expr = Tll(ti_b, ti_i) * R() * Su(ti_A);
  // \f$L^{a}{}_{bi} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LAbi_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(Tbi_R_SA_expr);
  // \f$L^{a}{}_{ib} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LAib_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(Tbi_R_SA_expr);
  // \f$L_{b}{}^{a}{}_{i} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LbAi_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(Tbi_R_SA_expr);
  // \f$L_{bi}{}^{a} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LbiA_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(Tbi_R_SA_expr);
  // \f$L_{i}{}^{a}{}_{b} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LiAb_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(Tbi_R_SA_expr);
  // \f$L_{ib}{}^{a} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LibA_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(Tbi_R_SA_expr);

  // \f$T_{bi} * S^{a} * R\f$
  const auto Tbi_SA_R_expr = Tll(ti_b, ti_i) * Su(ti_A) * R();
  // \f$L^{a}{}_{bi} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LAbi_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(Tbi_SA_R_expr);
  // \f$L^{a}{}_{ib} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LAib_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(Tbi_SA_R_expr);
  // \f$L_{b}{}^{a}{}_{i} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      LbAi_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(Tbi_SA_R_expr);
  // \f$L_{bi}{}^{a} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LbiA_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(Tbi_SA_R_expr);
  // \f$L_{i}{}^{a}{}_{b} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      LiAb_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(Tbi_SA_R_expr);
  // \f$L_{ib}{}^{a} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      LibA_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(Tbi_SA_R_expr);

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      for (size_t i = 0; i < 4; i++) {
        const DataType expected_R_SA_Tbi_product =
            R.get() * Su.get(a) * Tll.get(b, i);
        CHECK(LAbi_from_R_SA_Tbi.get(a, b, i) == expected_R_SA_Tbi_product);
        CHECK(LAib_from_R_SA_Tbi.get(a, i, b) == expected_R_SA_Tbi_product);
        CHECK(LbAi_from_R_SA_Tbi.get(b, a, i) == expected_R_SA_Tbi_product);
        CHECK(LbiA_from_R_SA_Tbi.get(b, i, a) == expected_R_SA_Tbi_product);
        CHECK(LiAb_from_R_SA_Tbi.get(i, a, b) == expected_R_SA_Tbi_product);
        CHECK(LibA_from_R_SA_Tbi.get(i, b, a) == expected_R_SA_Tbi_product);

        const DataType expected_R_Tbi_SA_product =
            R.get() * Tll.get(b, i) * Su.get(a);
        CHECK(LAbi_from_R_Tbi_SA.get(a, b, i) == expected_R_Tbi_SA_product);
        CHECK(LAib_from_R_Tbi_SA.get(a, i, b) == expected_R_Tbi_SA_product);
        CHECK(LbAi_from_R_Tbi_SA.get(b, a, i) == expected_R_Tbi_SA_product);
        CHECK(LbiA_from_R_Tbi_SA.get(b, i, a) == expected_R_Tbi_SA_product);
        CHECK(LiAb_from_R_Tbi_SA.get(i, a, b) == expected_R_Tbi_SA_product);
        CHECK(LibA_from_R_Tbi_SA.get(i, b, a) == expected_R_Tbi_SA_product);

        const DataType expected_SA_R_Tbi_product =
            Su.get(a) * R.get() * Tll.get(b, i);
        CHECK(LAbi_from_SA_R_Tbi.get(a, b, i) == expected_SA_R_Tbi_product);
        CHECK(LAib_from_SA_R_Tbi.get(a, i, b) == expected_SA_R_Tbi_product);
        CHECK(LbAi_from_SA_R_Tbi.get(b, a, i) == expected_SA_R_Tbi_product);
        CHECK(LbiA_from_SA_R_Tbi.get(b, i, a) == expected_SA_R_Tbi_product);
        CHECK(LiAb_from_SA_R_Tbi.get(i, a, b) == expected_SA_R_Tbi_product);
        CHECK(LibA_from_SA_R_Tbi.get(i, b, a) == expected_SA_R_Tbi_product);

        const DataType expected_SA_Tbi_R_product =
            Su.get(a) * Tll.get(b, i) * R.get();
        CHECK(LAbi_from_SA_Tbi_R.get(a, b, i) == expected_SA_Tbi_R_product);
        CHECK(LAib_from_SA_Tbi_R.get(a, i, b) == expected_SA_Tbi_R_product);
        CHECK(LbAi_from_SA_Tbi_R.get(b, a, i) == expected_SA_Tbi_R_product);
        CHECK(LbiA_from_SA_Tbi_R.get(b, i, a) == expected_SA_Tbi_R_product);
        CHECK(LiAb_from_SA_Tbi_R.get(i, a, b) == expected_SA_Tbi_R_product);
        CHECK(LibA_from_SA_Tbi_R.get(i, b, a) == expected_SA_Tbi_R_product);

        const DataType expected_Tbi_R_SA_product =
            Tll.get(b, i) * R.get() * Su.get(a);
        CHECK(LAbi_from_Tbi_R_SA.get(a, b, i) == expected_Tbi_R_SA_product);
        CHECK(LAib_from_Tbi_R_SA.get(a, i, b) == expected_Tbi_R_SA_product);
        CHECK(LbAi_from_Tbi_R_SA.get(b, a, i) == expected_Tbi_R_SA_product);
        CHECK(LbiA_from_Tbi_R_SA.get(b, i, a) == expected_Tbi_R_SA_product);
        CHECK(LiAb_from_Tbi_R_SA.get(i, a, b) == expected_Tbi_R_SA_product);
        CHECK(LibA_from_Tbi_R_SA.get(i, b, a) == expected_Tbi_R_SA_product);

        const DataType expected_Tbi_SA_R_product =
            Tll.get(b, i) * Su.get(a) * R.get();
        CHECK(LAbi_from_Tbi_SA_R.get(a, b, i) == expected_Tbi_SA_R_product);
        CHECK(LAib_from_Tbi_SA_R.get(a, i, b) == expected_Tbi_SA_R_product);
        CHECK(LbAi_from_Tbi_SA_R.get(b, a, i) == expected_Tbi_SA_R_product);
        CHECK(LbiA_from_Tbi_SA_R.get(b, i, a) == expected_Tbi_SA_R_product);
        CHECK(LiAb_from_Tbi_SA_R.get(i, a, b) == expected_Tbi_SA_R_product);
        CHECK(LibA_from_Tbi_SA_R.get(i, b, a) == expected_Tbi_SA_R_product);
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the inner product of two rank 1 tensors is correctly evaluated
///
/// \details
/// The inner product cases tested are:
/// - (rank 0) = (upper rank 1) x (lower rank 1)
/// - (rank 0) = (lower rank 1) x (upper rank 1)
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_rank_1_inner_product(const DataType& used_for_size) noexcept {
  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ru(used_for_size);
  create_tensor(make_not_null(&Ru));

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sl(used_for_size);
  create_tensor(make_not_null(&Sl));

  // \f$L = R^{a} * S_{a}\f$
  const Tensor<DataType> L_from_RA_Sa =
      TensorExpressions::evaluate(Ru(ti_A) * Sl(ti_a));
  // \f$L = S_{a} * R^{a}\f$
  const Tensor<DataType> L_from_Sa_RA =
      TensorExpressions::evaluate(Sl(ti_a) * Ru(ti_A));

  DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t a = 0; a < 4; a++) {
    expected_sum += (Ru.get(a) * Sl.get(a));
  }
  CHECK(L_from_RA_Sa.get() == expected_sum);
  CHECK(L_from_Sa_RA.get() == expected_sum);
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the inner product of two rank 2 tensors is correctly evaluated
///
/// \details
/// The sixteen inner product cases all contract both pairs of indices of the
/// two tensor operands to a resulting rank 0 tensor. The sixteen cases are
/// permutations of the positions of contracted pairs and their valences.
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_rank_2_inner_product(const DataType& used_for_size) noexcept {
  using lower_spacetime_index = SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>;
  using upper_spacetime_index = SpacetimeIndex<3, UpLo::Up, Frame::Inertial>;

  Tensor<DataType, Symmetry<2, 1>,
         index_list<lower_spacetime_index, lower_spacetime_index>>
      Rll(used_for_size);
  create_tensor(make_not_null(&Rll));
  Tensor<DataType, Symmetry<2, 1>,
         index_list<upper_spacetime_index, upper_spacetime_index>>
      Ruu(used_for_size);
  create_tensor(make_not_null(&Ruu));
  Tensor<DataType, Symmetry<2, 1>,
         index_list<lower_spacetime_index, upper_spacetime_index>>
      Rlu(used_for_size);
  create_tensor(make_not_null(&Rlu));
  Tensor<DataType, Symmetry<2, 1>,
         index_list<upper_spacetime_index, lower_spacetime_index>>
      Rul(used_for_size);
  create_tensor(make_not_null(&Rul));

  // \f$L = R_{ab} * R^{ab}\f$
  const Tensor<DataType> L_abAB_product =
      TensorExpressions::evaluate(Rll(ti_a, ti_b) * Ruu(ti_A, ti_B));
  // \f$L = R_{ab} * R^{ba}\f$
  const Tensor<DataType> L_abBA_product =
      TensorExpressions::evaluate(Rll(ti_a, ti_b) * Ruu(ti_B, ti_A));
  // \f$L = R_{ba} * R^{ab}\f$
  const Tensor<DataType> L_baAB_product =
      TensorExpressions::evaluate(Rll(ti_b, ti_a) * Ruu(ti_A, ti_B));
  // \f$L = R_{ba} * R^{ba}\f$
  const Tensor<DataType> L_baBA_product =
      TensorExpressions::evaluate(Rll(ti_b, ti_a) * Ruu(ti_B, ti_A));
  // \f$L = R^{ab} * R_{ab}\f$
  const Tensor<DataType> L_ABab_product =
      TensorExpressions::evaluate(Ruu(ti_A, ti_B) * Rll(ti_a, ti_b));
  // \f$L = R^{ab} * R_{ba}\f$
  const Tensor<DataType> L_ABba_product =
      TensorExpressions::evaluate(Ruu(ti_A, ti_B) * Rll(ti_b, ti_a));
  // \f$L = R^{ba} * R_{ab}\f$
  const Tensor<DataType> L_BAab_product =
      TensorExpressions::evaluate(Ruu(ti_B, ti_A) * Rll(ti_a, ti_b));
  // \f$L = R^{ba} * R_{ba}\f$
  const Tensor<DataType> L_BAba_product =
      TensorExpressions::evaluate(Ruu(ti_B, ti_A) * Rll(ti_b, ti_a));
  // \f$L = R_{a}{}^{b} * R^{a}{}_{b}\f$
  const Tensor<DataType> L_aBAb_product =
      TensorExpressions::evaluate(Rlu(ti_a, ti_B) * Rul(ti_A, ti_b));
  // \f$L = R_{a}{}^{b} * R_{b}{}^{a}\f$
  const Tensor<DataType> L_aBbA_product =
      TensorExpressions::evaluate(Rlu(ti_a, ti_B) * Rlu(ti_b, ti_A));
  // \f$L = R^{b}{}_{a} * R^{a}{}_{b}\f$
  const Tensor<DataType> L_BaAb_product =
      TensorExpressions::evaluate(Rul(ti_B, ti_a) * Rul(ti_A, ti_b));
  // \f$L = R^{b}{}_{a} * R_{b}{}^{a}\f$
  const Tensor<DataType> L_BabA_product =
      TensorExpressions::evaluate(Rul(ti_B, ti_a) * Rlu(ti_b, ti_A));
  // \f$L = R^{a}{}_{b} * R_{a}{}^{b}\f$
  const Tensor<DataType> L_AbaB_product =
      TensorExpressions::evaluate(Rul(ti_A, ti_b) * Rlu(ti_a, ti_B));
  // \f$L = R^{a}{}_{b} * R^{b}{}_{a}\f$
  const Tensor<DataType> L_AbBa_product =
      TensorExpressions::evaluate(Rul(ti_A, ti_b) * Rul(ti_B, ti_a));
  // \f$L = R_{b}{}^{a} * R_{a}{}^{b}\f$
  const Tensor<DataType> L_bAaB_product =
      TensorExpressions::evaluate(Rlu(ti_b, ti_A) * Rlu(ti_a, ti_B));
  // \f$L = R_{b}{}^{a} * R^{b}{}_{a}\f$
  const Tensor<DataType> L_bABa_product =
      TensorExpressions::evaluate(Rlu(ti_b, ti_A) * Rul(ti_B, ti_a));

  DataType L_abAB_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_abBA_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_baAB_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_baBA_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_ABab_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_ABba_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_BAab_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_BAba_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_aBAb_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_AbaB_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_aBbA_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_BaAb_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_AbBa_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_BabA_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_bAaB_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_bABa_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_abAB_expected_product += (Rll.get(a, b) * Ruu.get(a, b));
      L_abBA_expected_product += (Rll.get(a, b) * Ruu.get(b, a));
      L_baAB_expected_product += (Rll.get(b, a) * Ruu.get(a, b));
      L_baBA_expected_product += (Rll.get(b, a) * Ruu.get(b, a));
      L_ABab_expected_product += (Ruu.get(a, b) * Rll.get(a, b));
      L_ABba_expected_product += (Ruu.get(a, b) * Rll.get(b, a));
      L_BAab_expected_product += (Ruu.get(b, a) * Rll.get(a, b));
      L_BAba_expected_product += (Ruu.get(b, a) * Rll.get(b, a));
      L_aBAb_expected_product += (Rlu.get(a, b) * Rul.get(a, b));
      L_AbaB_expected_product += (Rul.get(a, b) * Rlu.get(a, b));
      L_aBbA_expected_product += (Rlu.get(a, b) * Rlu.get(b, a));
      L_BaAb_expected_product += (Rlu.get(b, a) * Rul.get(a, b));
      L_AbBa_expected_product += (Rul.get(a, b) * Rul.get(b, a));
      L_BabA_expected_product += (Ruu.get(b, a) * Rll.get(b, a));
      L_bAaB_expected_product += (Ruu.get(b, a) * Rll.get(a, b));
      L_bABa_expected_product += (Ruu.get(b, a) * Rll.get(b, a));
    }
  }
  CHECK(L_abAB_product.get() == L_abAB_expected_product);
  CHECK(L_abBA_product.get() == L_abBA_expected_product);
  CHECK(L_baAB_product.get() == L_baAB_expected_product);
  CHECK(L_baBA_product.get() == L_baBA_expected_product);
  CHECK(L_ABab_product.get() == L_ABab_expected_product);
  CHECK(L_ABba_product.get() == L_ABba_expected_product);
  CHECK(L_BAab_product.get() == L_BAab_expected_product);
  CHECK(L_BAba_product.get() == L_BAba_expected_product);
  CHECK(L_aBAb_product.get() == L_aBAb_expected_product);
  CHECK(L_AbaB_product.get() == L_AbaB_expected_product);
  CHECK(L_aBbA_product.get() == L_aBbA_expected_product);
  CHECK(L_BaAb_product.get() == L_BaAb_expected_product);
  CHECK(L_AbBa_product.get() == L_AbBa_expected_product);
  CHECK(L_BabA_product.get() == L_BabA_expected_product);
  CHECK(L_bAaB_product.get() == L_bAaB_expected_product);
  CHECK(L_bABa_product.get() == L_bABa_expected_product);
}

template <typename DataType>
void test_products(const DataType& used_for_size) noexcept {
  test_rank_0_outer_product(used_for_size);
  test_rank_1_outer_product(used_for_size);
  test_rank_2_outer_product(used_for_size);
  test_ranks_0_1_2_outer_product(used_for_size);

  test_rank_1_inner_product(used_for_size);
  test_rank_2_inner_product(used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Product",
                  "[DataStructures][Unit]") {
  test_products(std::numeric_limits<double>::signaling_NaN());
  test_products(DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerProduct2By2",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rll{};
  std::iota(Rll.begin(), Rll.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sll{};
  std::iota(Sll.begin(), Sll.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ruu{};
  std::iota(Ruu.begin(), Ruu.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Suu{};
  std::iota(Suu.begin(), Suu.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Rlu{};
  std::iota(Rlu.begin(), Rlu.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Slu{};
  std::iota(Slu.begin(), Slu.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rul{};
  std::iota(Rul.begin(), Rul.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sul{};
  std::iota(Sul.begin(), Sul.end(), 0.0);

  auto L_abAB_product =
      TensorExpressions::evaluate(Rll(ti_a, ti_b) * Suu(ti_A, ti_B));
  double L_abAB_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_abAB_expected_sum += (Rll.get(a, b) * Suu.get(a, b));
    }
  }
  CHECK(L_abAB_product.get() == L_abAB_expected_sum);

  auto L_abBA_product =
      TensorExpressions::evaluate(Rll(ti_a, ti_b) * Suu(ti_B, ti_A));
  double L_abBA_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_abBA_expected_sum += (Rll.get(a, b) * Suu.get(b, a));
    }
  }
  CHECK(L_abBA_product.get() == L_abBA_expected_sum);

  auto L_baAB_product =
      TensorExpressions::evaluate(Rll(ti_b, ti_a) * Suu(ti_A, ti_B));
  double L_baAB_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_baAB_expected_sum += (Rll.get(b, a) * Suu.get(a, b));
    }
  }
  CHECK(L_baAB_product.get() == L_baAB_expected_sum);

  auto L_baBA_product =
      TensorExpressions::evaluate(Rll(ti_b, ti_a) * Suu(ti_B, ti_A));
  double L_baBA_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_baBA_expected_sum += (Rll.get(b, a) * Suu.get(b, a));
    }
  }
  CHECK(L_baBA_product.get() == L_baBA_expected_sum);

  auto L_ABab_product =
      TensorExpressions::evaluate(Ruu(ti_A, ti_B) * Sll(ti_a, ti_b));
  double L_ABab_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_ABab_expected_sum += (Ruu.get(a, b) * Sll.get(a, b));
    }
  }
  CHECK(L_ABab_product.get() == L_ABab_expected_sum);

  auto L_ABba_product =
      TensorExpressions::evaluate(Ruu(ti_A, ti_B) * Sll(ti_b, ti_a));
  double L_ABba_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_ABba_expected_sum += (Ruu.get(a, b) * Sll.get(b, a));
    }
  }
  CHECK(L_ABba_product.get() == L_ABba_expected_sum);

  auto L_BAab_product =
      TensorExpressions::evaluate(Ruu(ti_B, ti_A) * Sll(ti_a, ti_b));
  double L_BAab_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_BAab_expected_sum += (Ruu.get(b, a) * Sll.get(a, b));
    }
  }
  CHECK(L_BAab_product.get() == L_BAab_expected_sum);

  auto L_BAba_product =
      TensorExpressions::evaluate(Ruu(ti_B, ti_A) * Sll(ti_b, ti_a));
  double L_BAba_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_BAba_expected_sum += (Ruu.get(b, a) * Sll.get(b, a));
    }
  }
  CHECK(L_BAba_product.get() == L_BAba_expected_sum);

  auto L_aBAb_product =
      TensorExpressions::evaluate(Rlu(ti_a, ti_B) * Sul(ti_A, ti_b));
  double L_aBAb_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_aBAb_expected_sum += (Rlu.get(a, b) * Sul.get(a, b));
    }
  }
  CHECK(L_aBAb_product.get() == L_aBAb_expected_sum);

  auto L_AbaB_product =
      TensorExpressions::evaluate(Rul(ti_A, ti_b) * Slu(ti_a, ti_B));
  double L_AbaB_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_AbaB_expected_sum += (Rul.get(a, b) * Slu.get(a, b));
    }
  }
  CHECK(L_AbaB_product.get() == L_AbaB_expected_sum);

  auto L_aBbA_product =
      TensorExpressions::evaluate(Rlu(ti_a, ti_B) * Slu(ti_b, ti_A));
  double L_aBbA_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_aBbA_expected_sum += (Rlu.get(a, b) * Slu.get(b, a));
    }
  }
  CHECK(L_aBbA_product.get() == L_aBbA_expected_sum);

  auto L_BaAb_product =
      TensorExpressions::evaluate(Rul(ti_B, ti_a) * Sul(ti_A, ti_b));
  double L_BaAb_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_BaAb_expected_sum += (Rlu.get(b, a) * Sul.get(a, b));
    }
  }
  CHECK(L_BaAb_product.get() == L_BaAb_expected_sum);

  auto L_AbBa_product =
      TensorExpressions::evaluate(Rul(ti_A, ti_b) * Sul(ti_B, ti_a));
  double L_AbBa_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_AbBa_expected_sum += (Rul.get(a, b) * Sul.get(b, a));
    }
  }
  CHECK(L_AbBa_product.get() == L_AbBa_expected_sum);

  auto L_BabA_product =
      TensorExpressions::evaluate(Rul(ti_B, ti_a) * Slu(ti_b, ti_A));
  double L_BabA_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_BabA_expected_sum += (Ruu.get(b, a) * Sll.get(b, a));
    }
  }
  CHECK(L_BabA_product.get() == L_BabA_expected_sum);

  auto L_bAaB_product =
      TensorExpressions::evaluate(Rlu(ti_b, ti_A) * Slu(ti_a, ti_B));
  double L_bAaB_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_bAaB_expected_sum += (Ruu.get(b, a) * Sll.get(a, b));
    }
  }
  CHECK(L_bAaB_product.get() == L_bAaB_expected_sum);

  auto L_bABa_product =
      TensorExpressions::evaluate(Rlu(ti_b, ti_A) * Sul(ti_B, ti_a));
  double L_bABa_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_bABa_expected_sum += (Ruu.get(b, a) * Sll.get(b, a));
    }
  }
  CHECK(L_bABa_product.get() == L_bABa_expected_sum);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerProduct2By2By2",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rll{};
  std::iota(Rll.begin(), Rll.end(), 0.0);
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>>>
      Suu{};
  std::iota(Suu.begin(), Suu.end(), 0.0);
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Tll{};
  std::iota(Tll.begin(), Tll.end(), 0.0);

  auto L_klKLij_to_ij = TensorExpressions::evaluate<ti_i, ti_j>(
      Rll(ti_k, ti_l) * Suu(ti_K, ti_L) * Tll(ti_i, ti_j));

  double trace = 0.0;
  for (size_t k = 0; k < 3; k++) {
    for (size_t l = 0; l < 3; l++) {
      trace += (Rll.get(k, l) * Suu.get(k, l));
    }
  }

  CHECK(TensorExpressions::evaluate(Rll(ti_k, ti_l) * Suu(ti_K, ti_L)).get() ==
        trace);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      double expected_sum = 0.0;
      for (size_t k = 0; k < 3; k++) {
        for (size_t l = 0; l < 3; l++) {
          expected_sum += Rll.get(k, l) * Suu.get(k, l) * Tll.get(i, j);
        }
      }
      CHECK(expected_sum == (trace * Tll.get(i, j)));
    }
  }

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      CHECK(L_klKLij_to_ij.get(i, j) == (trace * Tll.get(i, j)));
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct1By2By1",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ru{};
  std::iota(Ru.begin(), Ru.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Slu{};
  std::iota(Slu.begin(), Slu.end(), 0.0);
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Tl{};
  std::iota(Tl.begin(), Tl.end(), 0.0);

  auto L_aBCd = TensorExpressions::evaluate<ti_a, ti_B, ti_C, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_aBdC = TensorExpressions::evaluate<ti_a, ti_B, ti_d, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_aCBd = TensorExpressions::evaluate<ti_a, ti_C, ti_B, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_aCdB = TensorExpressions::evaluate<ti_a, ti_C, ti_d, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_adBC = TensorExpressions::evaluate<ti_a, ti_d, ti_B, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_adCB = TensorExpressions::evaluate<ti_a, ti_d, ti_C, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));

  auto L_BaCd = TensorExpressions::evaluate<ti_B, ti_a, ti_C, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_BadC = TensorExpressions::evaluate<ti_B, ti_a, ti_d, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_BCad = TensorExpressions::evaluate<ti_B, ti_C, ti_a, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_BCda = TensorExpressions::evaluate<ti_B, ti_C, ti_d, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_BdaC = TensorExpressions::evaluate<ti_B, ti_d, ti_a, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_BdCa = TensorExpressions::evaluate<ti_B, ti_d, ti_C, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));

  auto L_CaBd = TensorExpressions::evaluate<ti_C, ti_a, ti_B, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_CadB = TensorExpressions::evaluate<ti_C, ti_a, ti_d, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_CBad = TensorExpressions::evaluate<ti_C, ti_B, ti_a, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_CBda = TensorExpressions::evaluate<ti_C, ti_B, ti_d, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_CdaB = TensorExpressions::evaluate<ti_C, ti_d, ti_a, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_CdBa = TensorExpressions::evaluate<ti_C, ti_d, ti_B, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));

  auto L_daBC = TensorExpressions::evaluate<ti_d, ti_a, ti_B, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_daCB = TensorExpressions::evaluate<ti_d, ti_a, ti_C, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_dBaC = TensorExpressions::evaluate<ti_d, ti_B, ti_a, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_dBCa = TensorExpressions::evaluate<ti_d, ti_B, ti_C, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_dCaB = TensorExpressions::evaluate<ti_d, ti_C, ti_a, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_dCBa = TensorExpressions::evaluate<ti_d, ti_C, ti_B, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      for (size_t c = 0; c < 4; c++) {
        for (size_t d = 0; d < 4; d++) {
          double expected_product = Ru.get(b) * Slu.get(d, c) * Tl.get(a);
          CHECK(L_aBCd.get(a, b, c, d) == expected_product);
          CHECK(L_aBdC.get(a, b, d, c) == expected_product);
          CHECK(L_aCBd.get(a, c, b, d) == expected_product);
          CHECK(L_aCdB.get(a, c, d, b) == expected_product);
          CHECK(L_adBC.get(a, d, b, c) == expected_product);
          CHECK(L_adCB.get(a, d, c, b) == expected_product);

          CHECK(L_BaCd.get(b, a, c, d) == expected_product);
          CHECK(L_BadC.get(b, a, d, c) == expected_product);
          CHECK(L_BCad.get(b, c, a, d) == expected_product);
          CHECK(L_BCda.get(b, c, d, a) == expected_product);
          CHECK(L_BdaC.get(b, d, a, c) == expected_product);
          CHECK(L_BdCa.get(b, d, c, a) == expected_product);

          CHECK(L_CaBd.get(c, a, b, d) == expected_product);
          CHECK(L_CadB.get(c, a, d, b) == expected_product);
          CHECK(L_CBad.get(c, b, a, d) == expected_product);
          CHECK(L_CBda.get(c, b, d, a) == expected_product);
          CHECK(L_CdaB.get(c, d, a, b) == expected_product);
          CHECK(L_CdBa.get(c, d, b, a) == expected_product);

          CHECK(L_daBC.get(d, a, b, c) == expected_product);
          CHECK(L_daCB.get(d, a, c, b) == expected_product);
          CHECK(L_dBaC.get(d, b, a, c) == expected_product);
          CHECK(L_dBCa.get(d, b, c, a) == expected_product);
          CHECK(L_dCaB.get(d, c, a, b) == expected_product);
          CHECK(L_dCBa.get(d, c, b, a) == expected_product);
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerOuterProduct",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rll{};
  std::iota(Rll.begin(), Rll.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sul{};
  std::iota(Sul.begin(), Sul.end(), 0.0);
  auto L_abBc_to_ac = TensorExpressions::evaluate<ti_a, ti_c>(
      Rll(ti_a, ti_b) * Sul(ti_B, ti_c));
  auto L_abBc_to_ca = TensorExpressions::evaluate<ti_c, ti_a>(
      Rll(ti_a, ti_b) * Sul(ti_B, ti_c));

  for (size_t a = 0; a < 4; a++) {
    for (size_t c = 0; c < 4; c++) {
      double expected_sum = 0.0;
      for (size_t b = 0; b < 4; b++) {
        expected_sum += (Rll.get(a, b) * Sul.get(b, c));
      }
      CHECK(L_abBc_to_ac.get(a, c) == expected_sum);
      CHECK(L_abBc_to_ca.get(c, a) == expected_sum);
    }
  }

  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ru{};
  std::iota(Ru.begin(), Ru.end(), 0.0);
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sl{};
  std::iota(Sl.begin(), Sl.end(), 0.0);
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Tl{};
  std::iota(Tl.begin(), Tl.end(), 0.0);

  auto L_Aab_to_b =
      TensorExpressions::evaluate<ti_b>(Ru(ti_A) * Sl(ti_a) * Tl(ti_b));

  for (size_t b = 0; b < 4; b++) {
    double expected_sum = 0.0;
    for (size_t a = 0; a < 4; a++) {
      expected_sum += (Ru.get(a) * Sl.get(a) * Tl.get(b));
    }
    CHECK(L_Aab_to_b.get(b) == expected_sum);
  }

  Tensor<double, Symmetry<1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Tll{};
  std::iota(Tll.begin(), Tll.end(), 0.0);

  auto L_Aabc_to_bc = TensorExpressions::evaluate<ti_b, ti_c>(
      Ru(ti_A) * Sl(ti_a) * Tll(ti_b, ti_c));

  for (size_t c = 0; c < 4; c++) {
    for (size_t b = 0; b < 4; b++) {
      double expected_sum = 0.0;
      for (size_t a = 0; a < 4; a++) {
        expected_sum += (Ru.get(a) * Sl.get(a) * Tll.get(b, c));
      }
      CHECK(L_Aabc_to_bc.get(b, c) == expected_sum);
    }
  }
}
