// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRankN.hpp"

namespace {
// \brief Test evaluation of tensors where generic spatial indices are used for
// RHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs() {
  // test RHS Symmetry<2, 1>
  TestHelpers::tenex::test_evaluate<
      true, ti::a, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::i, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::A, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::a, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::I, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::i, ti::A, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::A, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::I, ti::A, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::i, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::i, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::i, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::J, ti::k, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::J, ti::k, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::J, ti::k, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::l, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::l, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::l, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::K, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::K, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::K, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();

  // test RHS Symmetry<1, 1> to LHS Symmetry<1, 1> and <2, 1>
  TestHelpers::tenex::test_evaluate<
      true, ti::a, ti::i, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::i, ti::a, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::A, ti::I, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::I, ti::A, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::J, ti::I, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Distorted>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Distorted>>,
      Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Distorted>,
                 SpatialIndex<3, UpLo::Up, Frame::Distorted>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::J, ti::I, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Distorted>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Distorted>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Distorted>,
                 SpatialIndex<3, UpLo::Up, Frame::Distorted>>>();

  TestHelpers::tenex::test_evaluate<
      true, ti::j, ti::k, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::j, ti::k, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
}

// \brief Test evaluation of tensors where generic spatial indices are used for
// LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_lhs() {
  // test RHS Symmetry<2, 1>
  TestHelpers::tenex::test_evaluate<
      false, ti::a, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::A, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::a, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::A, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::I, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::A, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::I, ti::A, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::J, ti::k, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::J, ti::k, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::J, ti::k, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::l, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::K, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::K, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::K, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  // test RHS Symmetry<1, 1> to LHS Symmetry<1, 1> and <2, 1>
  TestHelpers::tenex::test_evaluate<
      false, ti::K, ti::J, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::K, ti::J, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::K, ti::J, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::K, ti::J, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::l, ti::k, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::l, ti::k, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::l, ti::k, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::l, ti::k, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();
}

// \brief Test evaluation of rank 2 tensors where generic spatial indices are
// used for RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs_and_lhs_rank2() {
  // test RHS Symmetry<2, 1>

  // - two lower spatial tensor indices
  // - RHS one or two spacetime indices
  // - LHS two spacetime indices
  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::n, ti::m, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::m, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // - upper spatial tensor index, lower spatial tensor index
  // - RHS two spacetime indices
  // - LHS one or two spacetime indices
  TestHelpers::tenex::test_evaluate<
      false, ti::I, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::K, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::N, ti::k, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // - lower spatial tensor indices
  // - RHS spatial index, spacetime index
  // - LHS spatial index, spacetime index
  TestHelpers::tenex::test_evaluate<
      false, ti::j, ti::k, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // - upper spatial tensor indices
  // - RHS spacetime index, spatial index
  // - LHS spatial index, spacetime index
  TestHelpers::tenex::test_evaluate<
      false, ti::L, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();

  // - lower spatial tensor index, upper spatial tensor index
  // - RHS spatial index, spacetime index
  // - LHS spacetime index, spatial index
  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::M, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();

  // - lower spatial tensor indices
  // - RHS spacetime index, spatial index
  // - LHS spatial index, spacetime index
  TestHelpers::tenex::test_evaluate<
      false, ti::k, ti::j, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // - upper spatial tensor indices
  // - RHS spatial index, spacetime index
  // - LHS spacetime index, spatial index
  TestHelpers::tenex::test_evaluate<
      false, ti::L, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();

  // - lower spacetime tensor index, lower spatial tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate<
      false, ti::a, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // - lower spatial tensor index, lower spacetime tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // - upper spacetime tensor index, upper spatial tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate<
      false, ti::A, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  // - upper spatial tensor index, upper spacetime tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate<
      false, ti::I, ti::A, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  // - lower spatial tensor index, upper spacetime tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::A, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  // - lower spacetime tensor index, upper spatial tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate<
      false, ti::a, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  // - upper spatial tensor index, lower spacetime tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate<
      false, ti::I, ti::a, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // - upper spacetime tensor index, lower spatial tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate<
      false, ti::A, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // test RHS Symmetry<1, 1> to LHS Symmetry<1, 1> and <2, 1>

  TestHelpers::tenex::test_evaluate<
      false, ti::a, ti::i, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::a, ti::i, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::A, ti::I, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::A, ti::I, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::a, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::a, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::I, ti::A, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::I, ti::A, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::j, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::j, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate<
      false, ti::J, ti::I, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::J, ti::I, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<2, 1>>();
}

// \brief Test evaluation of rank 4 tensors where generic spatial indices are
// used for RHS and LHS spacetime indices
template <typename DataType>
void test_rhs_and_lhs_rank4() {
  // tests that return type is what is expected
  TestHelpers::tenex::test_evaluate<
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
  TestHelpers::tenex::test_evaluate<
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

// \brief Test evaluation of tensors where generic spatial indices are used for
// spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_spatial_spacetime_index() {
  test_rhs<DataType>();
  test_lhs<DataType>();
  test_rhs_and_lhs_rank2<DataType>();
  test_rhs_and_lhs_rank4<DataType>();
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateSpatialSpacetimeIndex",
    "[DataStructures][Unit]") {
  test_evaluate_spatial_spacetime_index<double>();
  test_evaluate_spatial_spacetime_index<DataVector>();
}
