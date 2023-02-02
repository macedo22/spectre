// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <set>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
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
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
//#include "Helpers/Domain/Creators/TestHelpers.hpp"
//#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Literals.hpp"

namespace {
// template <size_t Dim>
// void test_cost_function(
//   const std::vector<Block<Dim>>& blocks,
//     const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
//     const std::vector<std::array<size_t, Dim>>& initial_extents,
//     const Spectral::Quadrature quadrature) {
//   // print_costs_by_element_by_block(cost_by_element_by_block);
//   const domain::WeightedBlockZCurveProcDistribution<Dim> element_distribution{
//       number_of_procs_with_elements, cost_by_element_by_block,
//       global_procs_to_ignore};
//   // // std::cout << element_distribution.block_element_distribution() <<
//   // // std::endl;
//   // print_element_distribution(element_distribution.block_element_distribution());
// }

template <size_t Dim>
auto make_domain_creator(const std::string& opt_string) {
    return TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<Dim>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                Dim, domain::creators::AlignedLattice<Dim>>>(
        opt_string);
}

void test_cost_function() {
  // const size_t grid_points = 4;
  // // l_111 refers to refinement level 1 in x, y, and z
  // const std::array<size_t, 3> l_111{1, 1, 1}; // 8 subcells
  // const std::array<size_t, 3> l_112{1, 1, 2}; // 16 subcells
  // const std::array<size_t, 3> l_322{3, 2, 2}; // 128 subcells

  // std::vector<std::array<size_t, 3>> refinements_by_block(3);
  // refinements_by_block[0] = l_111;
  // refinements_by_block[0] = l_112;
  // refinements_by_block[0] = l_322;

  // const std::array<double, 3> block1_lower_xyz = {0.0, 0.0, 0.0};
  // const std::array<double, 3> block1_upper_xyz = {1.0, 1.0, 1.0};
  // const std::array<size_t, 3> block1_refinements = {2, 2, 1};
  // const std::array<size_t, 3> block1_grid_points = {1, 2, 3};
  // const std::array<bool, 3> block1_is_periodic = {{false, false, false}};

  // domain::creators::Brick brick1(
  //     block1_lower_xyz, block1_upper_xyz, block1_refinements,
  //     block1_grid_points, block1_is_periodic);
  
  // const auto domain1 = brick1.create_domain();
  // const auto blocks1 = domain1.blocks();
  // const auto block1 = std::move(blocks1[0]);
  // const auto block1_initial_extents = brick1.initial_extents()[0];

  // const size_t num_of_procs_to_use = 3;
  // const std::unordered_set<size_t> procs_to_ignore{};
  // std::vector<std::array<size_t, Dim>> initial_refinement_levels

  // const domain::BlockZCurveProcDistribution<Dim> element_distribution{
  //     num_of_procs_to_use, initial_refinement_levels, procs_to_ignore};

  // test_z_curve_index<Dim>(brick.create_domain(), brick.initial_extents(),
  //                                Spectral::Quadrature::GaussLobatto,
  //                                initial_refinement_level_xyz,
  //                                num_of_procs_to_use, procs_to_ignore);

  const auto domain_creator1 = make_domain_creator<3>(
        "AlignedLattice:\n"
        "  BlockBounds: [[0.0, 1.0, 2.0], [0.0, 1.0], [0.0, 1.0]]\n" +
            std::string{"  IsPeriodicIn: [false, false, false]\n"} +
            "  InitialGridPoints: [4, 4, 4]\n"
            "  InitialLevels: [2, 1, 0]\n"
            "  RefinedLevels: []\n"
            "  RefinedGridPoints: []\n"
            "  BlocksToExclude: []\n");
    const auto* aligned_blocks_creator1 =
        dynamic_cast<const domain::creators::AlignedLattice<3>*>(
            domain_creator1.get());
  const auto domain1 = aligned_blocks_creator1->create_domain();
  const auto& blocks1 = domain1.blocks();

  const auto domain_creator2 = make_domain_creator<3>(
        "AlignedLattice:\n"
        "  BlockBounds: [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]\n" +
            std::string{"  IsPeriodicIn: [false, false, false]\n"} +
            "  InitialGridPoints: [4, 4, 4]\n"
            "  InitialLevels: [2, 3, 2]\n"
            "  RefinedLevels: []\n"
            "  RefinedGridPoints: []\n"
            "  BlocksToExclude: []\n");
    const auto* aligned_blocks_creator2 =
        dynamic_cast<const domain::creators::AlignedLattice<3>*>(
            domain_creator2.get());
  const auto domain2 = aligned_blocks_creator2->create_domain();
  const auto& blocks2 = domain2.blocks();

  // const std::array<double, 3> brick1_lower_xyz = {0.0, 0.0, 0.0};
  // const std::array<double, 3> brick1_upper_xyz = {1.0, 1.0, 1.0};
  // const std::array<size_t, 3> brick1_refinements = {2, 2, 2};
  // const std::array<size_t, 3> brick1_grid_points = {4, 4, 4};
  // const std::array<bool, 3> brick1_is_periodic = {{false, false, false}};

  // domain::creators::Brick brick1(
  //     brick1_lower_xyz, brick1_upper_xyz, brick1_refinements,
  //     brick1_grid_points, brick1_is_periodic);
  
  // const auto domain2 = brick1.create_domain();
  // const auto& blocks2 = domain2.blocks();

  const auto domain_creator3 = make_domain_creator<3>(
        "AlignedLattice:\n"
        "  BlockBounds: [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]\n" +
            std::string{"  IsPeriodicIn: [false, false, false]\n"} +
            "  InitialGridPoints: [4, 3, 2]\n"
            "  InitialLevels: [2, 1, 0]\n"
            "  RefinedLevels: []\n"
            "  RefinedGridPoints: []\n"
            "  BlocksToExclude: []\n");
    const auto* aligned_blocks_creator3 =
        dynamic_cast<const domain::creators::AlignedLattice<3>*>(
            domain_creator3.get());
  const auto domain3 = aligned_blocks_creator3->create_domain();
  const auto& blocks3 = domain2.blocks();

  const auto costs1 =
      domain::WeightedBlockZCurveProcDistribution<3>::get_cost_by_element_by_block(
    blocks1,
    aligned_blocks_creator1->initial_refinement_levels(),
    aligned_blocks_creator1->initial_extents(),
    Spectral::Quadrature::GaussLobatto);

  const auto costs2 =
      domain::WeightedBlockZCurveProcDistribution<3>::get_cost_by_element_by_block(
    blocks2,
    aligned_blocks_creator2->initial_refinement_levels(),
    aligned_blocks_creator2->initial_extents(),
    Spectral::Quadrature::GaussLobatto);

  // const auto costs2 =
  //     domain::WeightedBlockZCurveProcDistribution<3>::get_cost_by_element_by_block(
  //   blocks2,
  //   brick1.initial_refinement_levels(),
  //   brick1.initial_extents(),
  //   Spectral::Quadrature::GaussLobatto);

  const auto costs3 =
      domain::WeightedBlockZCurveProcDistribution<3>::get_cost_by_element_by_block(
    blocks3,
    aligned_blocks_creator3->initial_refinement_levels(),
    aligned_blocks_creator3->initial_extents(),
    Spectral::Quadrature::GaussLobatto);

  // CHECK(costs1[0] == 2.0 * costs1[1]);
  // Approx custom_approx_e14 = Approx::custom().epsilon(1.0e-16).scale(1.0);
  // for (size_t i = 0; i < costs1[0].size(); i++) {
    // CHECK(costs1[0][i] == custom_approx_e14(costs1[1][i]));
    // CHECK(costs1[0][i] == costs1[1][i]);
  // }

  Approx custom_approx = Approx::custom().epsilon(1.0e-14).scale(1.0);

  const double elemental_cost1 = costs1[0][0];
  for (size_t i = 0; i < costs1[0].size(); i++) {
    // Approx custom_approx = Approx::custom().epsilon(1.0e-16).scale(1.0);
    CHECK(elemental_cost1 == custom_approx(costs1[0][i]));
    CHECK(elemental_cost1 == custom_approx(costs1[1][i]));
  }
  CHECK_ITERABLE_APPROX(costs1[0], costs1[1]);

  // CHECK(costs1[1] == 2.0 * costs1[2]);
  // Approx custom_approx = Approx::custom().epsilon(1.0e-14).scale(1.0);
  // for (size_t i = 0; i < costs1[0].size(); i++) {
  //   CHECK(costs1[0][i] == custom_approx(costs2[0][i]));
  //   // CHECK(costs1[0][i] == custom_approx(costs2[0][i]));
  // }
  // CHECK_ITERABLE_APPROX(costs1[0], costs2[0]);

  const double elemental_cost2 = costs2[0][0];
  for (size_t i = 1; i < costs2[0].size(); i++) {
    // Approx custom_approx = Approx::custom().epsilon(1.0e-16).scale(1.0);
    CHECK(elemental_cost2 == custom_approx(costs2[0][i]));
  }

  CHECK(elemental_cost2 == custom_approx(sqrt(2.0) * elemental_cost1));

  const double elemental_cost3 = costs3[0][0];
  for (size_t i = 1; i < costs3[0].size(); i++) {
    // Approx custom_approx = Approx::custom().epsilon(1.0e-16).scale(1.0);
    CHECK(elemental_cost3 == custom_approx(costs3[0][i]));
  }

  CHECK(elemental_cost3 == custom_approx(elemental_cost1 * 3.0 / 8.0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.WeightedElementDistribution", "[Domain][Unit]") {
  test_cost_function();

  // test<3>(10, get_uniform_cost(6, 4, 1.0));

  // const size_t Dim = 3;

  // const double inner_radius = 10.0;
  // const double outer_radius = 110.0;
  // const size_t initial_refinement = 1;
  // const std::array<size_t, 2> initial_number_of_grid_points{{5, 7}};
  // const bool use_equiangular_map = false;

  // domain::creators::Sphere sphere(
  //     inner_radius, outer_radius, initial_refinement,
  //     initial_number_of_grid_points, use_equiangular_map);

// const std::array<double, 3> lower_xyz = {0.0, 0.0, 0.0};
  // const std::array<double, 3> upper_xyz = {1.0, 10.0, 100.0};
  // const std::array<size_t, 3> initial_refinement_level_xyz = {1, 2, 3};
  // const std::array<size_t, 3> initial_number_of_grid_points_in_xyz = {2, 4, 6};
  // const std::array<bool, 3> is_periodic_in_xyz = {{false, false, false}};  

  // domain::creators::Brick brick(
  //     lower_xyz, upper_xyz, initial_refinement_level_xyz,
  //     initial_number_of_grid_points_in_xyz, is_periodic_in_xyz);

  // const size_t num_of_procs_to_use = 3;
  // const std::unordered_set<size_t> procs_to_ignore{};
  // // std::vector<std::array<size_t, Dim>> initial_refinement_levels

  // // const domain::BlockZCurveProcDistribution<Dim> element_distribution{
  // //     num_of_procs_to_use, initial_refinement_levels, procs_to_ignore};

  // test_z_curve_index<Dim>(brick.create_domain(), brick.initial_extents(),
  //                                Spectral::Quadrature::GaussLobatto,
  //                                initial_refinement_level_xyz,
  //                                num_of_procs_to_use, procs_to_ignore);
}
