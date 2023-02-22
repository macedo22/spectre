// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementDistribution.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Block.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/ZCurve.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"

namespace domain {
template <size_t Dim>
double get_num_points_and_grid_spacing_cost(
    const ElementId<Dim>& element_id, const Block<Dim>& block,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Spectral::Quadrature quadrature, const size_t num_grid_points) {
  Mesh<Dim> mesh = ::domain::Initialization::create_initial_mesh(
      initial_extents, element_id, quadrature);
  Element<Dim> element = ::domain::Initialization::create_initial_element(
      element_id, block, initial_refinement_levels);
  ElementMap<Dim, Frame::Grid> element_map{
      element_id, block.is_time_dependent()
                      ? block.moving_mesh_logical_to_grid_map().get_clone()
                      : block.stationary_map().get_to_grid_frame()};
  const tnsr::I<DataVector, Dim, Frame::ElementLogical> logical_coords =
      logical_coordinates(mesh);
  const tnsr::I<DataVector, Dim, Frame::Grid> grid_coords =
      element_map(logical_coords);
  const double min_grid_spacing =
      minimum_grid_spacing(mesh.extents(), grid_coords);

  return num_grid_points / sqrt(min_grid_spacing);
}

template <size_t Dim>
std::unordered_map<ElementId<Dim>, double> get_element_costs(
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const ElementWeight element_weight,
    const std::optional<Spectral::Quadrature>& quadrature) {
  std::unordered_map<ElementId<Dim>, double> element_costs{};

  for (size_t block_number = 0; block_number < blocks.size(); block_number++) {
    const auto& block = blocks[block_number];
    const auto initial_ref_levs = initial_refinement_levels[block_number];
    const std::vector<ElementId<Dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    const size_t grid_points_per_element = alg::accumulate(
        initial_extents[block_number], 1_st, std::multiplies<size_t>());

    for (const auto& element_id : element_ids) {
      if (element_weight == ElementWeight::Uniform) {
        element_costs.insert({element_id, 1.0});
      } else if (element_weight == ElementWeight::NumGridPoints) {
        element_costs.insert({element_id, grid_points_per_element});
      } else {
        ASSERT(quadrature.has_value(),
               "Since element_weight is "
               "ElementWeight::NumGridPointsAndGridSpacing, quadrature must "
               "have a value");
        const double cost = get_num_points_and_grid_spacing_cost(
            element_id, block, initial_refinement_levels, initial_extents,
            quadrature.value(), grid_points_per_element);

        element_costs.insert({element_id, cost});
      }
    }
  }

  return element_costs;
}

template <size_t Dim>
BlockZCurveProcDistribution<Dim>::BlockZCurveProcDistribution(
    const std::unordered_map<ElementId<Dim>, double>& element_costs,
    const size_t number_of_procs_with_elements,
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const std::unordered_set<size_t>& global_procs_to_ignore) {
  const size_t num_blocks = blocks.size();

  ASSERT(
      number_of_procs_with_elements > 0,
      "Must have a non-zero number of processors to distribute elements to.");
  ASSERT(num_blocks > 0, "Must have a non-zero number of blocks.");
  ASSERT(
      initial_refinement_levels.size() == num_blocks,
      "`initial_refinement_levels` is not the same size as number of blocks");
  ASSERT(initial_extents.size() == num_blocks,
         "`initial_extents` is not the same size as number of blocks");

  size_t num_elements = 0;
  std::vector<size_t> num_elements_by_block(num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    const size_t num_elements_current_block = two_to_the(alg::accumulate(
        initial_refinement_levels[i], 0_st, std::plus<size_t>()));
    num_elements_by_block[i] = num_elements_current_block;
    num_elements += num_elements_current_block;
  }

  ASSERT(element_costs.size() == num_elements,
         "`element_costs` is not the same size as the total number of elements "
         "computed from `initial_refinement_levels`");

  block_element_distribution_ =
      std::vector<std::vector<std::pair<size_t, size_t>>>(num_blocks);

  std::vector<std::vector<ElementId<Dim>>> initial_element_ids_by_block(
      num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    initial_element_ids_by_block[i].reserve(num_elements_by_block[i]);
    initial_element_ids_by_block[i] = initial_element_ids_in_z_curve_order(
        blocks[i].id(), initial_refinement_levels[i]);
  }

  double total_cost = 0.0;
  for (const auto& element_id_and_cost : element_costs) {
    total_cost += element_id_and_cost.second;
  }

  size_t current_block_num = 0;
  size_t element_num_of_block = 0;
  double cost_remaining = total_cost;
  size_t number_of_ignored_procs_so_far = 0;
  for (size_t i = 0; i < number_of_procs_with_elements; ++i) {
    size_t global_proc_number = i + number_of_ignored_procs_so_far;
    while (global_procs_to_ignore.find(global_proc_number) !=
           global_procs_to_ignore.end()) {
      ++number_of_ignored_procs_so_far;
      ++global_proc_number;
    }

    // The target cost per proc is updated as we distribute to each proc since
    // the total cost on a proc will nearly never be exactly the target average.
    // If we don't adjust the target cost, then we risk either not using all
    // procs (from overshooting the average too much on multiple procs) or
    // piling up cost on the last proc (from undershooting the average on
    // multiple procs). Updating the target cost per proc keeps the total cost
    // spread somewhat evenly to each proc.
    double target_cost_per_proc =
        cost_remaining / static_cast<double>(number_of_procs_with_elements - i);
    double cost_spent_on_proc = 0.0;
    size_t total_elements_distributed_to_proc = 0;
    bool add_more_elements_to_proc = true;
    // while we haven't yet distributed all blocks and we still have cost
    // allowed on the proc
    while (add_more_elements_to_proc and (current_block_num < num_blocks)) {
      const size_t num_elements_current_block =
          num_elements_by_block[current_block_num];
      size_t num_elements_distributed_to_proc = 0;
      // while we still have elements left on the block to distribute and we
      // still have cost allowed on the proc
      while (add_more_elements_to_proc and
             (element_num_of_block < num_elements_current_block)) {
        const ElementId<Dim>& element_id =
            initial_element_ids_by_block[current_block_num]
                                        [element_num_of_block];
        const double element_cost = element_costs.at(element_id);

        if (total_elements_distributed_to_proc == 0) {
          // if we haven't yet assigned any elements to this proc, assign the
          // current element to the current proc to ensure it gets at least one
          // element
          cost_remaining -= element_cost;
          cost_spent_on_proc = element_cost;
          num_elements_distributed_to_proc = 1;
          total_elements_distributed_to_proc = 1;
          element_num_of_block++;
        } else {
          const double current_cost_diff =
              abs(target_cost_per_proc - cost_spent_on_proc);
          const double next_cost_diff =
              abs(target_cost_per_proc - (cost_spent_on_proc + element_cost));

          if (current_cost_diff <= next_cost_diff) {
            // if the current proc cost is closer to the target cost than if we
            // were to add one more element, then we're done adding elements to
            // this proc and don't add the current one
            add_more_elements_to_proc = false;
          } else {
            // otherwise, the current proc cost is farther from the target then
            // if we were to add one more element, so we add the current element
            // to the current proc
            cost_spent_on_proc += element_cost;
            cost_remaining -= element_cost;
            num_elements_distributed_to_proc++;
            total_elements_distributed_to_proc++;
            element_num_of_block++;
          }
        }
      }

      // add a proc and its element allowance for the current block
      block_element_distribution_.at(current_block_num)
          .emplace_back(std::make_pair(global_proc_number,
                                       num_elements_distributed_to_proc));
      if (element_num_of_block >= num_elements_current_block) {
        // if we're done assigning elements from the current block, move on to
        // the elements in the next block
        ++current_block_num;
        element_num_of_block = 0;
      }
    }
  }
}

template <size_t Dim>
size_t BlockZCurveProcDistribution<Dim>::get_proc_for_element(
    const ElementId<Dim>& element_id) const {
  const size_t element_order_index = z_curve_index_from_element_id(element_id);
  size_t total_so_far = 0;
  for (const std::pair<size_t, size_t>& element_info :
       gsl::at(block_element_distribution_, element_id.block_id())) {
    if (total_so_far <= element_order_index and
        element_info.second + total_so_far > element_order_index) {
      return element_info.first;
    }
    total_so_far += element_info.second;
  }
  ERROR(
      "Processor not successfully chosen. This indicates a flaw in the logic "
      "of BlockZCurveProcDistribution.");
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                               \
  template class BlockZCurveProcDistribution<GET_DIM(data)>;                 \
  double get_num_points_and_grid_spacing_cost(                               \
      const ElementId<GET_DIM(data)>& element_id,                            \
      const Block<GET_DIM(data)>& block,                                     \
      const std::vector<std::array<size_t, GET_DIM(data)>>&                  \
          initial_refinement_levels,                                         \
      const std::vector<std::array<size_t, GET_DIM(data)>>& initial_extents, \
      Spectral::Quadrature quadrature, size_t num_grid_points);              \
  template std::unordered_map<ElementId<GET_DIM(data)>, double>              \
  get_element_costs(                                                         \
      const std::vector<Block<GET_DIM(data)>>& blocks,                       \
      const std::vector<std::array<size_t, GET_DIM(data)>>&                  \
          initial_refinement_levels,                                         \
      const std::vector<std::array<size_t, GET_DIM(data)>>& initial_extents, \
      ElementWeight element_weight,                                          \
      const std::optional<Spectral::Quadrature>& quadrature);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace domain
