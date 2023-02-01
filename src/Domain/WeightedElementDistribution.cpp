// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/WeightedElementDistribution.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Domain/ZCurve.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
template <size_t Dim>
WeightedBlockZCurveProcDistribution<Dim>::WeightedBlockZCurveProcDistribution(
    const size_t number_of_procs_with_elements,
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Spectral::Quadrature quadrature,
    const std::unordered_set<size_t>& global_procs_to_ignore) {
  const std::vector<std::vector<double>> cost_by_element_by_block =
      get_cost_by_element_by_block(blocks, initial_refinement_levels,
                                   initial_extents, quadrature);

  block_element_distribution_ =
      std::vector<std::vector<std::pair<size_t, size_t> > >(
          cost_by_element_by_block.size());

  double total_cost = 0.0;
  double min_cost = std::numeric_limits<double>::max();
  double max_cost = std::numeric_limits<double>::min();

  for (auto& block : cost_by_element_by_block) {
    const auto min_element_this_block = *alg::min_element(block);
    const auto max_element_this_block = *alg::max_element(block);
    if (min_element_this_block < min_cost) {
      min_cost = min_element_this_block;
    }
    if (max_element_this_block > max_cost) {
      max_cost = max_element_this_block;
    }

    for (double element_cost : block) {
      total_cost += element_cost;
    }
  }

  size_t current_block = 0;
  size_t current_element_of_current_block = 0;
  double cost_remaining = total_cost;
  // This variable will keep track of how many global procs we've skipped over
  // so far. This bookkeeping is necessary so the element gets placed on the
  // correct global proc. The loop variable `i` does not correspond to global
  // proc number. It's just an index
  size_t number_of_ignored_procs_so_far = 0;
  for (size_t i = 0; i < number_of_procs_with_elements; ++i) {
    size_t global_proc_number = i + number_of_ignored_procs_so_far;
    while (global_procs_to_ignore.find(global_proc_number) !=
           global_procs_to_ignore.end()) {
      ++number_of_ignored_procs_so_far;
      ++global_proc_number;
    }

    // initialize cost for this proc to be the current element
    // double cost_spent_on_proc =
    //     cost_by_element_by_block[current_block]
    //         [current_element_of_current_block];
    // size_t num_elements_distributed_to_proc = 1;
    double target_cost_per_proc =
        cost_remaining / (number_of_procs_with_elements - i);
    double cost_spent_on_proc = 0.0;
    size_t total_elements_distributed_to_proc = 0;
    bool add_more_elements_to_proc = true;
    const size_t num_blocks = cost_by_element_by_block.size();
    // while we still have cost allowed on the proc
    while (add_more_elements_to_proc and (current_block < num_blocks)) {
      const size_t num_elements_current_block =
          cost_by_element_by_block[current_block].size();
      // while we still have elements left on the block and we still
      // have cost allowed on the proc
      size_t num_elements_distributed_to_proc = 0;

      while (add_more_elements_to_proc and
             (current_element_of_current_block < num_elements_current_block)) {
        const double element_cost =
            cost_by_element_by_block[current_block]
                                    [current_element_of_current_block];

        if (total_elements_distributed_to_proc == 0) {
          cost_remaining -= element_cost;
          cost_spent_on_proc = element_cost;
          num_elements_distributed_to_proc = 1;
          total_elements_distributed_to_proc = 1;
          current_element_of_current_block++;
        } else {
          const double current_cost_diff =
              abs(target_cost_per_proc - cost_spent_on_proc);
          const double next_cost_diff =
              abs(target_cost_per_proc - (cost_spent_on_proc + element_cost));

          if (current_cost_diff <= next_cost_diff) {
            add_more_elements_to_proc = false;
          } else {
            cost_spent_on_proc += element_cost;
            cost_remaining -= element_cost;
            num_elements_distributed_to_proc++;
            total_elements_distributed_to_proc++;
            current_element_of_current_block++;
          }
        }
      }

      block_element_distribution_.at(current_block)
          .emplace_back(std::make_pair(global_proc_number,
                                       num_elements_distributed_to_proc));
      if (current_element_of_current_block >= num_elements_current_block) {
        ++current_block;
        current_element_of_current_block = 0;
      }
    }
  }
}

template <size_t Dim>
std::vector<std::vector<double>>
WeightedBlockZCurveProcDistribution<Dim>::get_cost_by_element_by_block(
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Spectral::Quadrature quadrature) {
  std::vector<std::vector<double>> cost_by_element_by_block(blocks.size());

  for (size_t block_number = 0; block_number < blocks.size(); block_number++) {
    const auto& block = blocks[block_number];
    const auto initial_ref_levs = initial_refinement_levels[block.id()];
    const std::vector<ElementId<Dim>> element_ids =
        initial_element_ids_in_z_curve_order(block.id(), initial_ref_levs);
    const size_t grid_points_per_element = alg::accumulate(
        initial_extents[block.id()], 1_st, std::multiplies<size_t>());

    cost_by_element_by_block[block_number].reserve(element_ids.size());

    for (const auto& element_id : element_ids) {
      // TODO : move this out of here, probably best to put it in
      // WeightedElementDistribution to keep all the logic handled there in
      // one class.
      Mesh<Dim> mesh = ::domain::Initialization::create_initial_mesh(
          initial_extents, element_id, quadrature);
      Element<Dim> element = ::domain::Initialization::create_initial_element(
          element_id, block, initial_refinement_levels);
      ElementMap<Dim, Frame::Grid> element_map{
          element_id, block.is_time_dependent()
                          ? block.moving_mesh_logical_to_grid_map().get_clone()
                          : block.stationary_map().get_to_grid_frame()};

      std::unique_ptr<
          ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>>
          grid_to_inertial_map;
      if (block.is_time_dependent()) {
        grid_to_inertial_map =
            block.moving_mesh_grid_to_inertial_map().get_clone();
      } else {
        grid_to_inertial_map =
            ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
                ::domain::CoordinateMaps::Identity<Dim>{});
      }

      tnsr::I<DataVector, Dim, Frame::ElementLogical> logical_coords{};
      domain::Tags::LogicalCoordinates<Dim>::function(
          make_not_null(&logical_coords), mesh);

      tnsr::I<DataVector, Dim, Frame::Grid> grid_coords{};
      domain::Tags::MappedCoordinates<
          domain::Tags::ElementMap<Dim, Frame::Grid>,
          domain::Tags::Coordinates<Dim, Frame::ElementLogical>>::
          function(make_not_null(&grid_coords), element_map, logical_coords);

      double minimum_grid_spacing =
          std::numeric_limits<double>::signaling_NaN();
      domain::Tags::MinimumGridSpacingCompute<Dim, Frame::Grid>::function(
          make_not_null(&minimum_grid_spacing), mesh, grid_coords);

      cost_by_element_by_block[block_number].emplace_back(
          grid_points_per_element / sqrt(minimum_grid_spacing));
    }
  }

  return cost_by_element_by_block;
}

template <size_t Dim>
size_t WeightedBlockZCurveProcDistribution<Dim>::get_proc_for_element(
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
      "of WeightedBlockZCurveProcDistribution.");
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class WeightedBlockZCurveProcDistribution<GET_DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace domain
