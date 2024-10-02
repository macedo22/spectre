// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression."
    "EvaluateSpatialAndTimeSpacetimeIndex",
    "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  TestHelpers::tenex::test_evaluate_rank_4<
      false, DataVector, Symmetry<3, 2, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::t, ti::a, ti::j, ti::i>(make_not_null(&generator));
}
