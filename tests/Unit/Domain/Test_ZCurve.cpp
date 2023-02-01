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
template <size_t Dim>
void test(const size_t block_id,
          const std::array<size_t, Dim>& block_refinements, size_t grid_index) {
  std::vector<ElementId<Dim>> element_ids_in_default_order =
      initial_element_ids(block_id, block_refinements, grid_index);
  //   std::vector<ElementId<Dim>> element_ids_in_z_curve_order =
  //   initial_element_ids_in_z_curve_order(
  //     block_id, block_refinements, grid_index);

  //   const std::vector<Block<Dim>>& blocks = domain.blocks();

  //   const std::vector<std::array<size_t, Dim>> initial_refinement_levels =
  //       get_initial_refinement_levels(blocks, initial_refinement_level_xyz);

  // size_t num_elements = two_to_the(initial_refinement_levels[0]);
  // for (size_t i = 1; i < Dim; i++) {
  //   num_elements *= two_to_the(initial_refinement_levels[1]);
  // }

  size_t num_elements = two_to_the(block_refinements[0]);
  for (size_t i = 1; i < Dim; i++) {
    num_elements *= two_to_the(block_refinements[i]);
  }

  // const auto element_distribution = get_element_distribution<Dim,
  // IsWeighted>(
  //     initial_refinement_levels, num_of_procs_to_use, procs_to_ignore);

  //   const domain::WeightedBlockZCurveProcDistribution<Dim>
  //         weighted_element_distribution(num_of_procs_to_use, domain.blocks(),
  //         initial_refinement_levels,
  //                              initial_extents, quadrature);

  for (const auto& element_id : element_ids_in_default_order) {
    // for (size_t j = 0; j < num_elements; j++) {
    //   const auto& element_id = element_ids[j];
    // std::cout << "element_id : " << element_id << std::endl;

    // const size_t target_proc =
    //     element_distribution.get_proc_for_element(element_id);

    const size_t result_z_order_index =
        domain::z_curve_index_from_element_id(element_id);
    // std::cout << "result_z_order_index : " << result_z_order_index <<
    // std::endl;
    const std::array<size_t, Dim> result_element_id =
        domain::element_id_from_z_curve_index(result_z_order_index,
                                              block_refinements);

    std::array<size_t, Dim> expected_element_id;
    for (size_t i = 0; i < Dim; ++i) {
      expected_element_id[i] = element_id.segment_id(i).index();
    }

    CHECK(result_element_id == expected_element_id);

    // // std::cout << "target_proc : " << target_proc << std::endl;
    // if (run > 1) break;
    // run++;
    // std::cout << std::endl;
  }
  const std::vector<ElementId<Dim>> element_ids_in_z_curve_order =
      initial_element_ids_in_z_curve_order(block_id, block_refinements);
  for (size_t i = 0; i < num_elements; i++) {
    const auto& element_id = element_ids_in_z_curve_order[i];
    CHECK(domain::z_curve_index_from_element_id(element_id) == i);
  }

  // TODO : check that the ElementIds preserve grid index?

  // std::cout << std::endl;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.ZCurve", "[Domain][Unit]") {
  // test<3>(10, get_uniform_cost(6, 4, 1.0));

  const size_t Dim = 3;

  // const double inner_radius = 10.0;
  // const double outer_radius = 110.0;
  // const size_t initial_refinement = 1;
  // const std::array<size_t, 2> initial_number_of_grid_points{{5, 7}};
  // const bool use_equiangular_map = false;

  // domain::creators::Sphere sphere(
  //     inner_radius, outer_radius, initial_refinement,
  //     initial_number_of_grid_points, use_equiangular_map);

  const std::array<double, 3> lower_xyz = {0.0, 0.0, 0.0};
  const std::array<double, 3> upper_xyz = {1.0, 10.0, 100.0};
  const std::array<size_t, 3> initial_refinement_level_xyz = {1, 2, 3};
  const std::array<size_t, 3> initial_number_of_grid_points_in_xyz = {2, 4, 6};
  const std::array<bool, 3> is_periodic_in_xyz = {{false, false, false}};

  domain::creators::Brick brick(
      lower_xyz, upper_xyz, initial_refinement_level_xyz,
      initial_number_of_grid_points_in_xyz, is_periodic_in_xyz);

  const size_t num_of_procs_to_use = 3;
  const std::unordered_set<size_t> procs_to_ignore{};
  // std::vector<std::array<size_t, Dim>> initial_refinement_levels

  // const domain::BlockZCurveProcDistribution<Dim> element_distribution{
  //     num_of_procs_to_use, initial_refinement_levels, procs_to_ignore};

  //   test_z_curve_index<Dim>(brick.create_domain(), brick.initial_extents(),
  //                                  Spectral::Quadrature::GaussLobatto,
  //                                  initial_refinement_level_xyz,
  //                                  num_of_procs_to_use, procs_to_ignore);

  test(1, initial_refinement_level_xyz, 2);
}
