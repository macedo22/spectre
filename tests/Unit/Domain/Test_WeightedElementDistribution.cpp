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
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/WeightedElementDistribution.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Rational.hpp"

namespace {
std::vector<std::vector<double>> get_uniform_cost(const size_t num_blocks,
                                                  const size_t num_elements,
                                                  const double cost) {
  std::vector<std::vector<double>> costs_by_element_by_block(num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    costs_by_element_by_block[i] = std::vector<double>(num_elements);
    for (size_t j = 0; j < num_elements; j++) {
      costs_by_element_by_block[i][j] = cost;
    }
  }
  return costs_by_element_by_block;
}

void print_costs_by_element_by_block(
    const std::vector<std::vector<double>>& costs_by_element_by_block) {
  std::cout << "Costs by element by block" << std::endl;
  for (size_t i = 0; i < costs_by_element_by_block.size(); i++) {
    std::cout << "Block " << i << ":\n\t{" << costs_by_element_by_block[i][0];
    for (size_t j = 1; j < costs_by_element_by_block[i].size(); j++) {
      std::cout << ", " << costs_by_element_by_block[i][j];
    }
    std::cout << "}" << std::endl;
  }
  std::cout << std::endl;
}

// std::vector<std::vector<double> > get_test_cost_2() {}

// std::vector<std::vector<double> > get_test_cost_3() {}

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

void test(const size_t number_of_procs_with_elements,
          const std::vector<std::vector<double>>& cost_by_element_by_block,
          const std::unordered_set<size_t>& global_procs_to_ignore = {}) {
  // print_costs_by_element_by_block(cost_by_element_by_block);
  const domain::WeightedBlockZCurveProcDistribution element_distribution{
      number_of_procs_with_elements, cost_by_element_by_block,
      global_procs_to_ignore};
  // // std::cout << element_distribution.block_element_distribution() <<
  // // std::endl;
  // print_element_distribution(element_distribution.block_element_distribution());
}

// for Sphere
template <size_t Dim>
std::vector<std::array<size_t, Dim>> get_initial_refinement_levels(
    const std::vector<Block<Dim>>& blocks, const size_t initial_refinement) {
  const size_t num_blocks = blocks.size();
  std::vector<std::array<size_t, Dim>> initial_refinement_levels(num_blocks);
  std::array<size_t, Dim> refinement_levels{};
  std::fill(refinement_levels.begin(), refinement_levels.end(),
            initial_refinement);

  for (size_t i = 0; i < num_blocks; i++) {
    initial_refinement_levels[i] = refinement_levels;
  }

  return initial_refinement_levels;
}

// for Brick
template <size_t Dim>
std::vector<std::array<size_t, Dim>> get_initial_refinement_levels(
    const std::vector<Block<Dim>>& blocks,
    const std::array<size_t, 3>& initial_refinement_level_xyz) {
  const size_t num_blocks = blocks.size();
  std::vector<std::array<size_t, Dim>> initial_refinement_levels(num_blocks);

  for (size_t i = 0; i < num_blocks; i++) {
    initial_refinement_levels[i] = initial_refinement_level_xyz;
  }

  return initial_refinement_levels;
}

template <size_t Dim, bool IsWeighted = true,
          typename ElementDistribution = tmpl::conditional_t<
              IsWeighted, domain::WeightedBlockZCurveProcDistribution,
              domain::BlockZCurveProcDistribution<Dim>>>
ElementDistribution get_element_distribution(
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const size_t num_of_procs_to_use,
    const std::unordered_set<size_t>& procs_to_ignore) {
  return ElementDistribution{num_of_procs_to_use, initial_refinement_levels,
                             procs_to_ignore};
}

template <size_t Dim, bool IsWeighted = true>
void test_z_curve_index(
    const Domain<Dim>& domain,
    const std::array<size_t, 3>& initial_refinement_level_xyz,
    const size_t num_of_procs_to_use,
    const std::unordered_set<size_t>& procs_to_ignore) {
  const std::vector<Block<Dim>>& blocks = domain.blocks();

  const std::vector<std::array<size_t, Dim>> initial_refinement_levels =
      get_initial_refinement_levels(blocks, initial_refinement_level_xyz);

  const auto element_distribution = get_element_distribution<Dim, IsWeighted>(
      initial_refinement_levels, num_of_procs_to_use, procs_to_ignore);

  // size_t run = 0;
  for (const auto& block : blocks) {
    // std::cout << "block id : " << block.id() << std::endl;

    const auto initial_ref_levs = initial_refinement_levels[block.id()];
    const std::vector<ElementId<Dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    for (const auto& element_id : element_ids) {
      // std::cout << "element_id : " << element_id << std::endl;

      // const size_t target_proc =
      //     element_distribution.get_proc_for_element(element_id);

      const size_t result_z_order_index =
          domain::z_curve_index(element_id);
      // std::cout << "result_z_order_index : " << result_z_order_index << std::endl;
      const std::array<size_t, Dim> result_element_id =
          domain::element_id_from_z_curve_index(result_z_order_index, initial_ref_levs);
      
      std::array<size_t, Dim>
          expected_element_id;
      for (size_t i = 0; i < Dim; ++i) {
        expected_element_id[i] =
            element_id.segment_id(i).index();
      }

      CHECK(result_element_id == expected_element_id);

      // // std::cout << "target_proc : " << target_proc << std::endl;
      // if (run > 1) break;
      // run++;
      // std::cout << std::endl;
    }
    // std::cout << std::endl;
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.WeightedElementDistribution", "[Domain][Unit]") {
  // test(10, get_uniform_cost(6, 4, 1.0));

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

  test_z_curve_index<Dim, false>(brick.create_domain(),
                                 initial_refinement_level_xyz,
                                 num_of_procs_to_use, procs_to_ignore);
}
