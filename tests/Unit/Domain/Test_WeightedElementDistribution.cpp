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

#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
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
  print_costs_by_element_by_block(cost_by_element_by_block);
  const domain::WeightedBlockZCurveProcDistribution element_distribution{
      number_of_procs_with_elements, cost_by_element_by_block,
      global_procs_to_ignore};
  // std::cout << element_distribution.block_element_distribution() <<
  // std::endl;
  print_element_distribution(element_distribution.block_element_distribution());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.WeightedElementDistribution", "[Domain][Unit]") {
  test(10, get_uniform_cost(6, 4, 1.0));
}
