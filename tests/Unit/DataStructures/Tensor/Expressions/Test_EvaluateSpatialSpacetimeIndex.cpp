// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <random>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRankN.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// \brief Test evaluation of tensors where generic spatial indices are used for
// RHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs() {
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::i, ti::a, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::A, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::a, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::I, ti::a, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::i, ti::A, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>,
      ti::J, ti::k, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::l, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::K, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::a, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Distorted>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Distorted>>,
      ti::J, ti::I, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Distorted>,
                 SpatialIndex<3, UpLo::Up, Frame::Distorted>>>();

  TestHelpers::tenex::test_evaluate<
      true, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::j, ti::k, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
}

// \brief Test evaluation of tensors where generic spatial indices are used for
// LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_lhs() {
  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::A, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::a, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::a, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::I, ti::a, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::i, ti::A, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::J, ti::k, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::l, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::K, ti::I, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::J, ti::I, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>,
      ti::j, ti::k, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::J, ti::K, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::l, ti::k, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();
}

// \brief Test evaluation of rank 2 tensors where generic spatial indices are
// used for RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs_and_lhs_rank2() {
  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::i>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::a>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::J, ti::I>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::n, ti::m, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::I, ti::j, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::K, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::j, ti::k>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::L, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::i, ti::M, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::j, ti::k>();
}

// \brief Test evaluation of rank 4 tensors where generic spatial indices are
// used for RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs_and_lhs_rank4() {
  // tests that return type is what is expected
  TestHelpers::tenex::test_evaluate_rank_4_core<
      true, ti::i, ti::k, ti::a, ti::j, DataType, Symmetry<1, 2, 2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<1, 3, 2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();

  // tests that only spatial components are filled for LHS tensor arg
  TestHelpers::tenex::test_evaluate_rank_4_core<
      false, ti::m, ti::c, ti::i, ti::j, DataType, Symmetry<1, 1, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<1, 1, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
}

template <typename DataType>
void test_evaluate_spatial_spacetime_index(const DataType& /*meta*/) {
  test_rhs<DataType>();
  test_lhs<DataType>();
  test_rhs_and_lhs_rank2<DataType>();
  test_rhs_and_lhs_rank4<DataType>();
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateSpatialSpacetimeIndex",
    "[DataStructures][Unit]") {
  // Test evaluation of tensors where generic spatial indices are used for
  // spacetime indices
  test_evaluate_spatial_spacetime_index(
      std::numeric_limits<double>::signaling_NaN());
  test_evaluate_spatial_spacetime_index(
      std::complex<double>(std::numeric_limits<double>::signaling_NaN(),
                           std::numeric_limits<double>::signaling_NaN()));
  test_evaluate_spatial_spacetime_index(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
  test_evaluate_spatial_spacetime_index(
      ComplexDataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
