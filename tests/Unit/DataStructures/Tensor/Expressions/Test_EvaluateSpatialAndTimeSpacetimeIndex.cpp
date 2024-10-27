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
  // test_lhs<DataType>();
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
  test_evaluate_time_and_spatial_spacetime_index(
      std::complex<double>(std::numeric_limits<double>::signaling_NaN(),
                           std::numeric_limits<double>::signaling_NaN()));
  test_evaluate_time_and_spatial_spacetime_index(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
  test_evaluate_time_and_spatial_spacetime_index(
      ComplexDataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
