// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <set>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Domain/WeightedElementDistribution.hpp"
#include "Domain/ZCurve.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Literals.hpp"

namespace {
// template <size_t Dim>
// void test_z_curve_index_from_element_id(const ElementId<Dim>& element_id,
// const size_t expected_z_curve_index) {
//   CHECK(domain::z_curve_index_from_element_id(element_id) ==
//   expected_z_curve_index);
// }

template <size_t Dim>
void test_z_curve(const std::array<size_t, Dim>& block_refinement_levels) {
  // The Z-curve does not depend on the block ID or grid index, so the choice of
  // these two values for this test is arbitrary
  const size_t block_id = 1;
  const size_t grid_index = 2;

  std::vector<ElementId<Dim>> element_ids_in_default_order =
      initial_element_ids(block_id, block_refinement_levels, grid_index);

  const size_t num_elements = element_ids_in_default_order.size();
  // Whether or not we have encountered certain result Z-curve indices
  // when transforming ElementIds of a block to their Z-curve indices
  std::vector<bool> z_curve_index_hit(num_elements);
  std::fill(z_curve_index_hit.begin(), z_curve_index_hit.end(), false);

  // Check that computing the Z-curve index from an ElementId and then
  // Segment indices from the computed Z-curve index matches the Segment indices
  // of the original ElementId (i.e. transforming to and back from Z-curve index
  // recovers the original Segment indices of an ElementId). This checks that
  // domain::z_curve_index_from_element_i` and
  // domain::segment_indices_from_z_curve_index are consistent with each other
  // in that they are inverses.
  for (const auto& element_id : element_ids_in_default_order) {
    const size_t result_z_curve_index =
        domain::z_curve_index_from_element_id(element_id);
    const std::array<size_t, Dim> result_segment_indices =
        domain::segment_indices_from_z_curve_index(result_z_curve_index,
                                                   block_refinement_levels);

    std::array<size_t, Dim> expected_segment_indices;
    for (size_t i = 0; i < Dim; ++i) {
      expected_segment_indices[i] = element_id.segment_id(i).index();
    }

    CHECK(result_segment_indices == expected_segment_indices);

    z_curve_index_hit[result_z_curve_index] = true;
  }

  // Check that there is a 1:1 mapping of ElementId to Z-curve index
  for (const size_t index_hit : z_curve_index_hit) {
    CHECK(index_hit);
  }

  const std::vector<ElementId<Dim>> element_ids_in_z_curve_order =
      initial_element_ids_in_z_curve_order(block_id, block_refinement_levels,
                                           grid_index);

  // Check that there is a 1:1 mapping of ElementIds in the original input list
  // to the output list of ElementIds in Z-curve order (i.e. we preserved the
  // input list of ElementIds)
  CHECK(element_ids_in_z_curve_order.size() == num_elements);
  for (size_t i = 0; i < num_elements; i++) {
    const ElementId<Dim> element_id_from_z_curve_order =
        element_ids_in_z_curve_order[i];
    bool element_id_found = false;
    for (size_t j = 0; j < num_elements; j++) {
      const ElementId<Dim> element_id_from_default_order =
          element_ids_in_default_order[j];
      if (element_id_from_z_curve_order == element_id_from_default_order) {
        element_id_found = true;
        break;
      }
    }
    CHECK(element_id_found);
  }

  // Check that the ElementIds are ordered by Z-curve index
  for (size_t i = 0; i < element_ids_in_z_curve_order.size(); i++) {
    const auto& element_id = element_ids_in_z_curve_order[i];
    CHECK(domain::z_curve_index_from_element_id(element_id) == i);
  }
}

// Test Z-curve for each refinement level permutation for dimensions 1, 2, and 3
void test_z_curve_for_refinement_levels(const size_t min_refinement_level,
                                        const size_t max_refinement_level) {
  std::array<size_t, 1> block_refinement_levels_1d{};
  std::array<size_t, 2> block_refinement_levels_2d{};
  std::array<size_t, 3> block_refinement_levels_3d{};

  for (size_t i = min_refinement_level; i < max_refinement_level + 1; i++) {
    block_refinement_levels_1d[0] = i;
    block_refinement_levels_2d[0] = i;
    block_refinement_levels_3d[0] = i;

    test_z_curve(block_refinement_levels_1d);

    for (size_t j = min_refinement_level; j < max_refinement_level + 1; j++) {
      block_refinement_levels_2d[1] = j;
      block_refinement_levels_3d[1] = j;

      test_z_curve(block_refinement_levels_2d);

      for (size_t k = min_refinement_level; k < max_refinement_level + 1; k++) {
        block_refinement_levels_3d[2] = k;

        test_z_curve(block_refinement_levels_3d);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.ZCurve", "[Domain][Unit]") {
  const size_t min_refinement_level = 0;
  const size_t max_refinement_level = 3;

  test_z_curve_for_refinement_levels(min_refinement_level,
                                     max_refinement_level);
}
