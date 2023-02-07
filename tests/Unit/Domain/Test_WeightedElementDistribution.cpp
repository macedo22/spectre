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
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Protocols/Metavariables.hpp"
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
#include "Utilities/MakeArray.hpp"
#include "Utilities/Literals.hpp"

namespace {
// void print_costs_by_element_by_block(
//     const std::vector<std::vector<double>>& costs_by_element_by_block) {
//   std::cout << "Costs by element by block" << std::endl;
//   for (size_t i = 0; i < costs_by_element_by_block.size(); i++) {
//     std::cout << "Block " << i << ":\n\t[" << costs_by_element_by_block[i][0];
//     for (size_t j = 1; j < costs_by_element_by_block[i].size(); j++) {
//       std::cout << ", " << costs_by_element_by_block[i][j];
//     }
//     std::cout << "]" << std::endl;
//   }
//   std::cout << std::endl;
// }

void print_element_distribution(
    const std::vector<std::vector<std::pair<size_t, size_t>>>&
        block_element_distribution) {
  std::cout << "Block element distribution" << std::endl;
  for (size_t i = 0; i < block_element_distribution.size(); i++) {
    std::cout << "Block " << i << ":\n\t{{"
              << block_element_distribution[i][0].first << ", "
              << block_element_distribution[i][0].second << "}";
    for (size_t j = 1; j < block_element_distribution[i].size(); j++) {
      std::cout << ", {" << block_element_distribution[i][j].first << ", "
                << block_element_distribution[i][j].second << "}";
    }
    std::cout << "}" << std::endl;
  }
  std::cout << std::endl;
}

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

template <size_t Dim, bool EnableTimeDependentMaps, bool WithBoundaryConditions>
struct Metavariables {
  struct domain : tt::ConformsTo<::domain::protocols::Metavariables> {
    static constexpr bool enable_time_dependent_maps = EnableTimeDependentMaps;
  };
  using system = tmpl::conditional_t<WithBoundaryConditions,
                                     TestHelpers::domain::BoundaryConditions::
                                         SystemWithBoundaryConditions<Dim>,
                                     TestHelpers::domain::BoundaryConditions::
                                         SystemWithoutBoundaryConditions<Dim>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        DomainCreator<3>, tmpl::list<::domain::creators::BinaryCompactObject>>>;
  };
};

std::string stringize(const bool t) { return t ? "true" : "false"; }

std::string create_option_string(const bool excise_A, const bool excise_B,
                                 const bool add_time_dependence,
                                 const bool use_logarithmic_map_AB,
                                 const size_t additional_refinement_outer,
                                 const size_t additional_refinement_A,
                                 const size_t additional_refinement_B,
                                 const bool add_boundary_condition) {
  const std::string time_dependence{
      add_time_dependence ? "  TimeDependentMaps:\n"
                            "    InitialTime: 1.0\n"
                            "    ExpansionMap: \n"
                            "      OuterBoundary: 25.0\n"
                            "      InitialExpansion: 1.0\n"
                            "      InitialExpansionVelocity: -0.1\n"
                            "      AsymptoticVelocityOuterBoundary: -0.1\n"
                            "      DecayTimescaleOuterBoundaryVelocity: 5.0\n"
                            "    RotationMap:\n"
                            "      InitialAngularVelocity: [0.0, 0.0, -0.2]\n"
                            "    SizeMap:\n"
                            "      InitialValues: [0.0, 0.0]\n"
                            "      InitialVelocities: [-0.1, -0.2]\n"
                            "      InitialAccelerations: [0.01, 0.02]"
                          : ""};
  const std::string interior_A{
      add_boundary_condition
          ? std::string{"    Interior:\n" +
                        std::string{excise_A
                                        ? "      ExciseWithBoundaryCondition:\n"
                                          "        TestBoundaryCondition:\n"
                                          "          Direction: lower-zeta\n"
                                          "          BlockId: 50\n"
                                        : "      Auto\n"}}
          : "    ExciseInterior: " + stringize(excise_A) + "\n"};
  const std::string interior_B{
      add_boundary_condition
          ? std::string{"    Interior:\n" +
                        std::string{excise_B
                                        ? "      ExciseWithBoundaryCondition:\n"
                                          "        TestBoundaryCondition:\n"
                                          "          Direction: lower-zeta\n"
                                          "          BlockId: 50\n"
                                        : "      Auto\n"}}
          : "    ExciseInterior: " + stringize(excise_B) + "\n"};
  const std::string outer_boundary_condition{
      add_boundary_condition ? std::string{"    BoundaryCondition:\n"
                                           "      TestBoundaryCondition:\n"
                                           "        Direction: upper-zeta\n"
                                           "        BlockId: 50\n"}
                             : ""};
  return "BinaryCompactObject:\n"
         "  ObjectA:\n"
         "    InnerRadius: 1.0\n"
         "    OuterRadius: 2.0\n"
         "    XCoord: 3.0\n" +
         interior_A +
         "    UseLogarithmicMap: " + stringize(use_logarithmic_map_AB) +
         "\n"
         "  ObjectB:\n"
         "    InnerRadius: 0.2\n"
         "    OuterRadius: 1.0\n"
         "    XCoord: -2.0\n" +
         interior_B +
         "    UseLogarithmicMap: " + stringize(use_logarithmic_map_AB) +
         "\n"
         "  EnvelopingCube:\n"
         "    Radius: 22.0\n"
         "    UseProjectiveMap: true\n"
         "    Sphericity: 1.0\n"
         "  OuterShell:\n"
         "    InnerRadius: Auto\n"
         "    OuterRadius: 25.0\n"
         "    RadialDistribution: Linear\n" +
         outer_boundary_condition + "  InitialRefinement:\n" +
         (excise_A ? "" : "    ObjectAInterior: [1, 1, 1]\n") +
         (excise_B ? "" : "    ObjectBInterior: [1, 1, 1]\n") +
         "    ObjectAShell: [1, 1, " +
         std::to_string(1 + additional_refinement_A) +
         "]\n"
         "    ObjectBShell: [1, 1, " +
         std::to_string(1 + additional_refinement_B) +
         "]\n"
         "    ObjectACube: [1, 1, 1]\n"
         "    ObjectBCube: [1, 1, 1]\n"
         "    EnvelopingCube: [1, 1, 1]\n"
         "    OuterShell: [1, 1, " +
         std::to_string(1 + additional_refinement_outer) +
         "]\n"
         "  InitialGridPoints: 3\n" +
         time_dependence;
}

void test_cost_function() {
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

template <size_t Dim>
void test_element_distribution(
  const DomainCreator<Dim>& domain_creator,
  const size_t number_of_procs_with_elements,
  const std::unordered_set<size_t>& global_procs_to_ignore = {}) {
  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto initial_refinement_levels = domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();

  // TODO : run this test for different proc # and procs to skip?;
  // const size_t number_of_procs_with_elements = 73;
  // const std::unordered_set<size_t> global_procs_to_ignore{{5, 8, 9, number_of_procs_with_elements + 2}};
  const domain::WeightedBlockZCurveProcDistribution<Dim> element_distribution(
    number_of_procs_with_elements,
    blocks,
    initial_refinement_levels,
    initial_extents,
    Spectral::Quadrature::GaussLobatto,
    global_procs_to_ignore
  );

  const auto proc_map = element_distribution.block_element_distribution();
  // std::cout << "proc map : " << std::endl;
  // print_element_distribution(proc_map);

  // std::cout << "[";
  // for (size_t i = 0; i < proc_map.size(); i++) {
  //   std::cout << "[" << proc_map[i][0];
  //   for (size_t j = 1 j < proc_map[i].size(); j++) {
  //     std::cout << ", " << proc_map[i][j];
  //   }
  //   std::cout << "]";
  // }
  // std::cout << "]" << std::endl;

  size_t num_elements = 0;

  std::vector<std::vector<ElementId<Dim>>> element_ids_in_z_curve_order(blocks.size());
  for (size_t i = 0; i < blocks.size(); i++) {
    // const std::vector<ElementId<Dim>> element_ids_in_z_curve_order =
    //   domain::initial_element_ids_in_z_curve_order(
    //       block_id, block_refinement_levels, grid_index);
    element_ids_in_z_curve_order[i] = domain::initial_element_ids_in_z_curve_order(
          i, initial_refinement_levels[i], 0);
    num_elements += element_ids_in_z_curve_order[i].size();
  }

  const auto costs =
      domain::WeightedBlockZCurveProcDistribution<Dim>::get_cost_by_element_by_block(
    blocks,
    initial_refinement_levels,
    initial_extents,
    Spectral::Quadrature::GaussLobatto);
  
  // CHECK(costs.size() == num_elements);

  double total_cost = 0.0;
  for (const auto& block : costs) {
    for (const double element_cost : block) {
      total_cost += element_cost;
    }
  }

  std::vector<double> costs_flattened(num_elements);
  size_t cost_index = 0;
  for (const auto& block : costs) {
    for (const double element_cost : block) {
      costs_flattened[cost_index] = element_cost;
      cost_index++;
    }
  }

  // std::cout << "costs_flattened : " << costs_flattened << std::endl;

  // double average = total_cost / number_of_procs_with_elements;
  // double current_cost = 0.0;
  // size_t 

  const size_t total_procs = number_of_procs_with_elements + global_procs_to_ignore.size();
  // auto costs_by_proc =
  //     make_array<total_procs, double>(0.0);
  // std::vector<double> costs_by_proc(total_procs);
  // std::fill(costs_by_proc.begin(), costs_by_proc.end(), 0.0);
  std::vector<size_t> num_elements_each_proc(total_procs);
  std::fill(num_elements_each_proc.begin(), num_elements_each_proc.end(), 0);
  // size_t cost_starting_index = 0;
  // for (const auto& proc_map_current_block : proc_map) {
  for (size_t block_number = 0; block_number < proc_map.size(); block_number++) {
    // size_t element_starting_index = 0;
    for (const auto& proc_allowance : proc_map[block_number]) {
      const size_t proc_number = proc_allowance.first;
      const size_t element_allowance = proc_allowance.second;
      num_elements_each_proc[proc_number] += element_allowance;
      // for (size_t i = element_starting_index; i < element_starting_index + element_allowance; i++) {
      //   costs_by_proc[proc_number] += costs[block_number][i];
      // }
      // element_starting_index += element_allowance;
    }
  }

  // std::cout << "num_elements_each_proc : " << num_elements_each_proc << std::endl;

  // std::cout << costs_by_proc << std::endl;
  // std::cout << num_elements_each_proc << std::endl;

  // double average = total_cost / number_of_procs_with_elements;
  // double current_cost_this_proc = 0.0;

  // double cost_remaining = total_cost;
  // for (size_t i = 0; i < total_procs; i++) {
  //   const double target_proc_cost =
  //       cost_remaining / (number_of_procs_with_elements - i);
  //   double current_cost_this_proc = 0.0;
  //   size_t block_number = 0;
  //   size_t element_number = 0;

  //   const size_t element_allowance = num_elements_each_proc[i];
  //   while (block_number < blocks.size()) {
  //     current_cost_this_proc +=
  //     block_number++; 
  //   }
  // }

  cost_index = 0;

  double cost_remaining = total_cost;
  size_t procs_skipped = 0;
  for (size_t i = 0; i < total_procs; i++) {
    // std::cout << "global_proc_number : " << i << std::endl;
    // std::cout << "cost_index : " << cost_index << std::endl;
    // std::cout << "cost_remaining : " << cost_remaining << std::endl;
    // std::cout << "cost spent : " << (total_cost - cost_remaining) << std::endl;
    if (global_procs_to_ignore.count(i)) {
      //cost_index++;
      procs_skipped++;
      continue;
    }
    const double target_proc_cost =
        cost_remaining / (number_of_procs_with_elements - i + procs_skipped);
    // std::cout << "target_proc_cost: " << target_proc_cost << std::endl;
    double proc_cost_without_final_element = 0.0;
    const size_t num_elements_this_proc = num_elements_each_proc[i];
    // std::cout << "num_elements_this_proc : " << num_elements_this_proc << std::endl;
    if (num_elements_this_proc == 0) {
      continue;
    }
    // go to the element before the last one included
    for (size_t j = 0; j < num_elements_this_proc - 1; j++) {
      const double this_cost = costs_flattened[cost_index + j];
      // std::cout << "this_cost : " << this_cost << std::endl;
      proc_cost_without_final_element += this_cost;
    }
    // cost_index += num_elements_this_proc;
    // cost_remaining-=proc_cost_without_final_element;

    // std::cout << "proc_cost_without_final_element: " << proc_cost_without_final_element << std::endl;

    const double proc_cost_with_final_element =
      proc_cost_without_final_element + costs_flattened[cost_index + num_elements_this_proc - 1];

    //  std::cout << "proc_cost_with_final_element: " << proc_cost_with_final_element << std::endl;

    const double diff_without_final_element =
        abs(proc_cost_without_final_element - target_proc_cost);
    
    // std::cout << "diff_without_final_element: " << diff_without_final_element << std::endl;

    const double diff_with_final_element =
        abs(proc_cost_with_final_element - target_proc_cost);
    
    // std::cout << "diff_with_final_element: " << diff_with_final_element << std::endl;

    // if we've exceeded the target, make sure we're including this element because
    // it is closer to the target than if we don't include it
    if (num_elements_this_proc > 1 and proc_cost_with_final_element > target_proc_cost) {
      // std::cout << "proc_cost_with_final_element > target_proc_cost";
      CHECK(diff_with_final_element <= diff_without_final_element);
    }

    if (cost_index + num_elements_this_proc < num_elements) {
      const double proc_cost_with_extra_element =
      proc_cost_with_final_element + costs_flattened[cost_index + num_elements_this_proc];
      const double diff_with_extra_element =
        abs(proc_cost_with_extra_element - target_proc_cost);

        CHECK(diff_with_extra_element >= diff_with_final_element);
    }

    cost_index += num_elements_this_proc;
    cost_remaining-=proc_cost_with_final_element;

  }

  // cost_index = 0;

  // double cost_remaining = total_cost;
  // size_t procs_skipped = 0;
  // for (size_t i = 0; i < total_procs; i++) {
  //   if (global_procs_to_ignore.count(i)) {
  //     cost_index++;
  //     procs_skipped++;
  //     continue;
  //   }
  //   const double target_proc_cost =
  //       cost_remaining / (number_of_procs_with_elements - i + procs_skipped);
  //   double proc_cost_without_final_element = 0.0;
  //   const size_t num_elements_this_proc = num_elements_each_proc[i];
  //   size_t j = 0
  //   while (j < num_elements_this_proc - 1) {
  //     const double this_cost = costs_flattened[cost_index + j];
  //     proc_cost_without_final_element += this_cost;
  //   }
  //   // cost_index += num_elements_this_proc;
  //   // cost_remaining-=proc_cost_without_final_element;

  //   const double proc_cost_with_final_element =
  //     proc_cost_without_final_element + costs_flattened[cost_index + num_elements_this_proc - 1];
  //   CHECK(abs(proc_cost_without_final_element - target_proc_cost) >= abs(proc_cost_with_final_element - target_proc_cost));

  //   cost_index += num_elements_this_proc;
  //   cost_remaining-=proc_cost_with_final_element;

  // }
}

template <size_t Dim>
void test_proc_retrieval(
    const DomainCreator<Dim>& domain_creator,
  const size_t number_of_procs_with_elements,
  const std::unordered_set<size_t>& global_procs_to_ignore = {}
) {
  
  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto initial_refinement_levels = domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();

  const domain::WeightedBlockZCurveProcDistribution<Dim> element_distribution(
    number_of_procs_with_elements,
    blocks,
    initial_refinement_levels,
    initial_extents,
    Spectral::Quadrature::GaussLobatto,
    global_procs_to_ignore
  );

  size_t expected_total_num_elements = 0;
  const size_t num_blocks = blocks.size();

  std::vector<std::vector<ElementId<Dim>>> element_ids_in_z_curve_order(blocks.size());
  std::vector<size_t> expected_num_elements_by_block(num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    element_ids_in_z_curve_order[i] = domain::initial_element_ids_in_z_curve_order(
          i, initial_refinement_levels[i], 0);
    // std::cout << "initial_refinement_levels for this block : " << initial_refinement_levels[i][0] << std::endl;
    expected_num_elements_by_block[i] = 1;
    for (size_t j = 0; j < Dim; j++) {
      expected_num_elements_by_block[i] *= two_to_the(initial_refinement_levels[i][j]);
    }
    expected_total_num_elements += expected_num_elements_by_block[i];
  }

  const auto proc_map = element_distribution.block_element_distribution();

  std::cout << "num_blocks : " << blocks.size() << std::endl;
  std::cout << "expected_total_num_elements : " << expected_total_num_elements << std::endl;

  print_element_distribution(proc_map);

  const size_t total_number_of_procs =
      number_of_procs_with_elements + global_procs_to_ignore.size();
  
  std::vector<bool> proc_hit(total_number_of_procs);
  std::fill(proc_hit.begin(), proc_hit.end(), false);

  size_t actual_total_num_elements = 0;
  size_t highest_proc_assigned = 0;
  for (size_t i = 0; i < blocks.size(); i++) {
    size_t element_index = 0;
    const size_t expected_num_elements_this_block = expected_num_elements_by_block[i];
    const std::vector<std::pair<size_t, size_t>>& proc_map_this_block =
        proc_map[i];
    const size_t num_procs_this_block = proc_map_this_block.size();
    size_t actual_num_elements_this_block = 0;

    for (size_t j = 0; j < num_procs_this_block; j++) {
      const size_t expected_proc = proc_map_this_block[j].first;
      const size_t proc_allowance = proc_map_this_block[j].second;

      if (highest_proc_assigned < expected_proc) {
        highest_proc_assigned = expected_proc;
      }

      for (size_t k = 0; k < proc_allowance; k++) {
        CHECK(element_distribution.get_proc_for_element(
          element_ids_in_z_curve_order[i][element_index]) == expected_proc);
      }
      proc_hit[expected_proc] = true;
      element_index += proc_allowance;
      actual_num_elements_this_block += proc_allowance;
    }
    CHECK(actual_num_elements_this_block == expected_num_elements_this_block);
    actual_total_num_elements += actual_num_elements_this_block;
  }
  CHECK(actual_total_num_elements == expected_total_num_elements);

  for (size_t i = 0; i < highest_proc_assigned + 1; i++) {
    if (global_procs_to_ignore.count(i) == 0) {
      CHECK(proc_hit[i]);
    } else {
      CHECK(not proc_hit[i]);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.WeightedElementDistribution", "[Domain][Unit]") {
  test_cost_function();
  
  // Test inputs

  // 1D, single block
  const auto lattice_1d = make_domain_creator<1>(
        "AlignedLattice:\n"
        "  BlockBounds: [[0.0, 1.0]]\n" +
            std::string{"  IsPeriodicIn: [false]\n"} +
            "  InitialGridPoints: [6]\n"
            "  InitialLevels: [4]\n"
            "  RefinedLevels: []\n"
            "  RefinedGridPoints: []\n"
            "  BlocksToExclude: []\n");
  
  // 2D
  const auto lattice_2d = make_domain_creator<2>(
        "AlignedLattice:\n"
        "  BlockBounds: [[0.0, 0.3], [0.0, 0.8, 2.5, 4.9]]\n" +
            std::string{"  IsPeriodicIn: [false, false]\n"} +
            "  InitialGridPoints: [4, 5]\n"
            "  InitialLevels: [2, 3]\n"
            "  RefinedLevels: []\n"
            "  RefinedGridPoints: []\n"
            "  BlocksToExclude: []\n");

  // 3D
  const auto binary_compact_object_creator =
     TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                          Metavariables<3, true, false>>(
          create_option_string(true, true, true, false, 0, 0, 0,
                               false));

  // Test element distribution for 1D, 2D, 3D with and without procs to ignore
  test_element_distribution(*lattice_1d, 1);
  test_element_distribution(*lattice_1d, 5);
  test_element_distribution(*lattice_1d, 33, std::unordered_set<size_t>{7});

  test_element_distribution(*lattice_2d, 1);
  test_element_distribution(*lattice_2d, 10);
  test_element_distribution(*lattice_2d, 54, std::unordered_set<size_t>{0, 1});

  test_element_distribution(*binary_compact_object_creator, 1);
  test_element_distribution(*binary_compact_object_creator, 12);
  test_element_distribution(*binary_compact_object_creator, 73, std::unordered_set<size_t>{5, 8, 9, 75});

  // Test proc retrieval for 1D, 2D, 3D with and without procs to ignore
  test_proc_retrieval(*lattice_1d, 1);
  test_proc_retrieval(*lattice_1d, 5);
  test_proc_retrieval(*lattice_1d, 33, std::unordered_set<size_t>{7});

  test_proc_retrieval(*lattice_2d, 1);
  test_proc_retrieval(*lattice_2d, 10);
  test_proc_retrieval(*lattice_2d, 54, std::unordered_set<size_t>{0, 1});

  test_proc_retrieval(*binary_compact_object_creator, 1);
  test_proc_retrieval(*binary_compact_object_creator, 12);
  test_proc_retrieval(*binary_compact_object_creator, 73, std::unordered_set<size_t>{5, 8, 9, 75});
}
