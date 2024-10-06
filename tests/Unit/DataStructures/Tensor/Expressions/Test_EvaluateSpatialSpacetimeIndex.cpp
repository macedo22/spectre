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
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// \brief Test evaluation of tensors where generic spatial indices are used for
// RHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs() {
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, ti::a, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, ti::i, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, ti::i, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, ti::K, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_core<
      true, ti::a, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_core<
      true, ti::i, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_core<
      false, ti::J, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Distorted>,
                 SpatialIndex<3, UpLo::Up, Frame::Distorted>>,
      Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Distorted>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Distorted>>>();

  TestHelpers::tenex::test_evaluate_rank_2_core<
      true, ti::j, ti::k, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();
}

// \brief Test evaluation of tensors where generic spatial indices are used for
// LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_lhs() {
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::a, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::A, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::a, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::i, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::I, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::i, ti::A, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::i, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::J, ti::k, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::l, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::K, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_core<
      false, ti::J, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_core<
      false, ti::k, ti::l, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Grid>>,
      Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_core<
      false, ti::J, ti::K, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_core<
      false, ti::i, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
}

// \brief Test evaluation of rank 2 tensors where generic spatial indices are
// used for RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs_and_lhs_rank2() {
  // TODO : add more test cases here
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::a, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, ti::a, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::i, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, ti::i, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
}

// \brief Test evaluation of rank 4 tensors where generic spatial indices are
// used for RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs_and_lhs_rank4() {
  TestHelpers::tenex::test_evaluate_rank_4_core<
      true, ti::i, ti::k, ti::a, ti::j, DataType, Symmetry<4, 3, 2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<3, 2, 2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();
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
