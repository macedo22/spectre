// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <limits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename Generator, typename DataType>
void test_rhs(const gsl::not_null<Generator*> generator,
              const DataType& used_for_size) {
  // Note: this function doesn't utilize test helper functions like
  // test_evaluate_rank_2_core() because they aren't generic enough to handle
  // test cases where the number of indices on the RHS and LHS are not equal.
  // Instead, we have to manually check each test case of interest.

  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;
  using frame = Frame::Inertial;

  // RHS tensors with non-symmetric spacetime indices
  const auto R_ab = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_AB = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_Ab = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_aB = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);

  // RHS tensors with symmetric spacetime indices
  const auto S_ab = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto S_AB = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);

  // Evaluations of non-symmetric RHS tensors

  // \f$L_{i} = R_{it}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Lo, frame>>>
      L_i_from_R_it = tenex::evaluate<ti::i>(R_ab(ti::i, ti::t));
  // \f$L_{i} = R_{ti}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Lo, frame>>>
      L_i_from_R_ti = tenex::evaluate<ti::i>(R_ab(ti::t, ti::i));
  // \f$L^{i} = R^{it}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Up, frame>>>
      L_I_from_R_IT = tenex::evaluate<ti::I>(R_AB(ti::I, ti::T));
  // \f$L^{i} = R^{ti}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Up, frame>>>
      L_I_from_R_TI = tenex::evaluate<ti::I>(R_AB(ti::T, ti::I));
  // \f$L^{i} = R^{i}{}_{t}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Up, frame>>>
      L_I_from_R_It = tenex::evaluate<ti::I>(R_Ab(ti::I, ti::t));
  // \f$L_{i} = R^{t}{}_{i}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Lo, frame>>>
      L_i_from_R_Ti = tenex::evaluate<ti::i>(R_Ab(ti::T, ti::i));
  // \f$L_{i} = R_{i}{}^{t}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Lo, frame>>>
      L_i_from_R_iT = tenex::evaluate<ti::i>(R_aB(ti::i, ti::T));
  // \f$L^{i} = R_{t}{}^{i}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Up, frame>>>
      L_I_from_R_tI = tenex::evaluate<ti::I>(R_aB(ti::t, ti::I));

  // Evaluations of symmetric RHS tensors

  // \f$L_{i} = S_{it}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Lo, frame>>>
      L_i_from_S_it = tenex::evaluate<ti::i>(S_ab(ti::i, ti::t));
  // \f$L_{i} = S_{ti}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Lo, frame>>>
      L_i_from_S_ti = tenex::evaluate<ti::i>(S_ab(ti::t, ti::i));
  // \f$L^{i} = S^{it}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Up, frame>>>
      L_I_from_S_IT = tenex::evaluate<ti::I>(S_AB(ti::I, ti::T));
  // \f$L^{i} = S^{ti}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Up, frame>>>
      L_I_from_S_TI = tenex::evaluate<ti::I>(S_AB(ti::T, ti::I));

  for (size_t i = 0; i < dim; i++) {
    CHECK(L_i_from_R_it.get(i) == R_ab.get(i + 1, 0));
    CHECK(L_i_from_R_ti.get(i) == R_ab.get(0, i + 1));
    CHECK(L_I_from_R_IT.get(i) == R_AB.get(i + 1, 0));
    CHECK(L_I_from_R_TI.get(i) == R_AB.get(0, i + 1));
    CHECK(L_I_from_R_It.get(i) == R_Ab.get(i + 1, 0));
    CHECK(L_i_from_R_Ti.get(i) == R_Ab.get(0, i + 1));
    CHECK(L_i_from_R_iT.get(i) == R_aB.get(i + 1, 0));
    CHECK(L_I_from_R_tI.get(i) == R_aB.get(0, i + 1));

    CHECK(L_i_from_S_it.get(i) == S_ab.get(i + 1, 0));
    CHECK(L_i_from_S_ti.get(i) == S_ab.get(0, i + 1));
    CHECK(L_I_from_S_IT.get(i) == S_AB.get(i + 1, 0));
    CHECK(L_I_from_S_TI.get(i) == S_AB.get(0, i + 1));
  }
}

template <typename Generator, typename DataType>
void test_lhs(const gsl::not_null<Generator*> generator,
              const DataType& used_for_size) {
  // Note: this function doesn't utilize test helper functions like
  // test_evaluate_rank_2_core() because they aren't generic enough to handle
  // test cases where the number of indices on the RHS and LHS are not equal.
  // Instead, we have to manually check each test case of interest.

  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;
  using frame = Frame::Inertial;

  const auto R =
      make_with_random_values<DataType>(generator, distribution, used_for_size);

  // Test evaluation of RHS scalar to non-symmetric LHS rank 2

  // \f$L_{it} = R\f$
  auto L_it_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::i, ti::t>(make_not_null(&L_it_from_R), R);
  // \f$L_{ti} = R\f$
  auto L_ti_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::i>(make_not_null(&L_ti_from_R), R);
  // \f$L^{it} = R\f$
  auto L_IT_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::I, ti::T>(make_not_null(&L_IT_from_R), R);
  // \f$L^{ti} = R\f$
  auto L_TI_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::I>(make_not_null(&L_TI_from_R), R);
  // \f$L^{i}{}_{t} = R\f$
  auto L_It_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::I, ti::t>(make_not_null(&L_It_from_R), R);
  // \f$L^{t}{}_{i} = R\f$
  auto L_Ti_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::i>(make_not_null(&L_Ti_from_R), R);
  // \f$L_{i}{}^{t} = R\f$
  auto L_iT_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::i, ti::T>(make_not_null(&L_iT_from_R), R);
  // \f$L_{t}{}^{i} = R\f$
  auto L_tI_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::I>(make_not_null(&L_tI_from_R), R);

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t b = 0; b < dim + 1; b++) {
      const auto expected_value =
          (a > 0 and b == 0) ? R
                             : TestHelpers::tenex::component_placeholder_value<
                                   DataType>::value;
      CHECK(L_it_from_R.get(a, b) == expected_value);
      CHECK(L_ti_from_R.get(b, a) == expected_value);
      CHECK(L_IT_from_R.get(a, b) == expected_value);
      CHECK(L_TI_from_R.get(b, a) == expected_value);
      CHECK(L_It_from_R.get(a, b) == expected_value);
      CHECK(L_Ti_from_R.get(b, a) == expected_value);
      CHECK(L_iT_from_R.get(a, b) == expected_value);
      CHECK(L_tI_from_R.get(b, a) == expected_value);
    }
  }

  // Test evaluation of RHS scalar to symmetric LHS rank 2

  // \f$M_{it} = R\f$
  auto M_it_from_R =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::i, ti::t>(make_not_null(&M_it_from_R), R);
  // \f$M_{ti} = R\f$
  auto M_ti_from_R =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::i>(make_not_null(&M_ti_from_R), R);
  // \f$M^{it} = R\f$
  auto M_IT_from_R =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::I, ti::T>(make_not_null(&M_IT_from_R), R);
  // \f$M^{ti} = R\f$
  auto M_TI_from_R =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::I>(make_not_null(&M_TI_from_R), R);

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t b = a; b < dim + 1; b++) {
      const auto expected_value =
          (a == 0 and b > 0) ? R
                             : TestHelpers::tenex::component_placeholder_value<
                                   DataType>::value;
      CHECK(M_it_from_R.get(a, b) == expected_value);
      CHECK(M_ti_from_R.get(b, a) == expected_value);
      CHECK(M_IT_from_R.get(a, b) == expected_value);
      CHECK(M_TI_from_R.get(b, a) == expected_value);
    }
  }

  // RHS Rank 1

  const auto R_i = make_with_random_values<tnsr::i<DataType, dim, frame>>(
      generator, distribution, used_for_size);
  const auto R_I = make_with_random_values<tnsr::I<DataType, dim, frame>>(
      generator, distribution, used_for_size);

  // Test evaluation of RHS rank 1 to non-symmetric LHS rank 2

  // \f$L_{it} = R_i\f$
  auto L_it_from_R_i =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::i, ti::t>(make_not_null(&L_it_from_R_i), R_i(ti::i));
  // \f$L_{ti} = R_i\f$
  auto L_ti_from_R_i =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::i>(make_not_null(&L_ti_from_R_i), R_i(ti::i));
  // \f$L^{it} = R^i\f$
  auto L_IT_from_R_I =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::I, ti::T>(make_not_null(&L_IT_from_R_I), R_I(ti::I));
  // \f$L^{ti} = R^i\f$
  auto L_TI_from_R_I =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::I>(make_not_null(&L_TI_from_R_I), R_I(ti::I));
  // \f$L^{i}{}_{t} = R^i\f$
  auto L_It_from_R_I =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::I, ti::t>(make_not_null(&L_It_from_R_I), R_I(ti::I));
  // \f$L^{t}{}_{i} = R_i\f$
  auto L_Ti_from_R_i =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::i>(make_not_null(&L_Ti_from_R_i), R_i(ti::i));
  // \f$L_{i}{}^{t} = R_i\f$
  auto L_iT_from_R_i =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::i, ti::T>(make_not_null(&L_iT_from_R_i), R_i(ti::i));
  // \f$L_{t}{}^{i} = R^i\f$
  auto L_tI_from_R_I =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::I>(make_not_null(&L_tI_from_R_I), R_I(ti::I));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t b = 0; b < dim + 1; b++) {
      if (a > 0 and b == 0) {
        CHECK(L_it_from_R_i.get(a, b) == R_i.get(a - 1));
        CHECK(L_ti_from_R_i.get(b, a) == R_i.get(a - 1));
        CHECK(L_IT_from_R_I.get(a, b) == R_I.get(a - 1));
        CHECK(L_TI_from_R_I.get(b, a) == R_I.get(a - 1));
        CHECK(L_It_from_R_I.get(a, b) == R_I.get(a - 1));
        CHECK(L_Ti_from_R_i.get(b, a) == R_i.get(a - 1));
        CHECK(L_iT_from_R_i.get(a, b) == R_i.get(a - 1));
        CHECK(L_tI_from_R_I.get(b, a) == R_I.get(a - 1));
      } else {
        CHECK(L_it_from_R_i.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(L_ti_from_R_i.get(b, a) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(L_IT_from_R_I.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(L_TI_from_R_I.get(b, a) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(L_It_from_R_I.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(L_Ti_from_R_i.get(b, a) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(L_iT_from_R_i.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(L_tI_from_R_I.get(b, a) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
      }
    }
  }

  // Test evaluation of RHS rank 1 to symmetric LHS rank 2

  // \f$M_{it} = R_i\f$
  auto M_it_from_R_i =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::i, ti::t>(make_not_null(&M_it_from_R_i), R_i(ti::i));
  // \f$M_{ti} = R_i\f$
  auto M_ti_from_R_i =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::i>(make_not_null(&M_ti_from_R_i), R_i(ti::i));
  // \f$M^{it} = R^i\f$
  auto M_IT_from_R_I =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::I, ti::T>(make_not_null(&M_IT_from_R_I), R_I(ti::I));
  // \f$M^{ti} = R^i\f$
  auto M_TI_from_R_I =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::I>(make_not_null(&M_TI_from_R_I), R_I(ti::I));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t b = a; b < dim + 1; b++) {
      if (a == 0 and b > 0) {
        CHECK(M_it_from_R_i.get(a, b) == R_i.get(b - 1));
        CHECK(M_ti_from_R_i.get(b, a) == R_i.get(b - 1));
        CHECK(M_IT_from_R_I.get(a, b) == R_I.get(b - 1));
        CHECK(M_TI_from_R_I.get(b, a) == R_I.get(b - 1));
      } else {
        CHECK(M_it_from_R_i.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(M_ti_from_R_i.get(b, a) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(M_IT_from_R_I.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(M_TI_from_R_I.get(b, a) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
      }
    }
  }
}

template <typename DataType>
void test_rhs_and_lhs_rank4() {
  using FrameType = Frame::Inertial;
  using symm_1111 = Symmetry<1, 1, 1, 1>;
  using index_list_abcd = index_list<SpacetimeIndex<3, UpLo::Lo, FrameType>,
                                     SpacetimeIndex<3, UpLo::Lo, FrameType>,
                                     SpacetimeIndex<3, UpLo::Lo, FrameType>,
                                     SpacetimeIndex<3, UpLo::Lo, FrameType>>;

  TestHelpers::tenex::test_evaluate_rank_4<false, ti::t, ti::a, ti::j, ti::i,
                                           symm_1111, index_list_abcd>();
}

template <typename DataType>
void test_evaluate_time_and_spatial_spacetime_index(
    const DataType& used_for_size) {
  MAKE_GENERATOR(generator);

  test_rhs(make_not_null(&generator), used_for_size);
  test_lhs(make_not_null(&generator), used_for_size);
  // test_rhs_and_lhs_rank2<DataType>();
  test_rhs_and_lhs_rank4<DataType>();
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression."
    "EvaluateSpatialAndTimeSpacetimeIndex",
    "[DataStructures][Unit]") {
  // Test evaluation of tensors where concrete time indices and spatial indices
  // are used for spacetime indices

  test_evaluate_time_and_spatial_spacetime_index(
      std::numeric_limits<double>::signaling_NaN());
  // test_evaluate_time_and_spatial_spacetime_index(
  //     std::complex<double>(std::numeric_limits<double>::signaling_NaN(),
  //                          std::numeric_limits<double>::signaling_NaN()));
  // test_evaluate_time_and_spatial_spacetime_index(
  //     DataVector(5, std::numeric_limits<double>::signaling_NaN()));
  // test_evaluate_time_and_spatial_spacetime_index(
  //     ComplexDataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
