// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Domain/Structure/ElementId.hpp"

namespace domain {

template <size_t Dim>
size_t z_curve_index(const ElementId<Dim>& element_id);

template <size_t Dim>
std::array<size_t, Dim> element_id_from_z_curve_index(
    const size_t z_order_index,
    const std::array<size_t, Dim>& block_refinements);

/*!
 * \brief Distribution strategy for assigning elements to CPUs using a
 * Morton ('Z-order') space-filling curve to determine placement within each
 * block.
 *
 * \details The element distribution assigns a balanced number of elements to
 * each processor that is allowed to have elements (default all). Specify which
 * processors aren't allowed to have elements by passing in an unordered set of
 * `size_t`s corresponding to the processor number. This distribution is
 * computed by first greedily assigning to each available processor an allowance
 * of [total number of elements]/[number of processors available] elements from
 * one or more blocks, starting with the lowest number block that still has
 * elements to contribute to an allowance. Then, once those allowances are
 * determined, a separate Z-order curve is established for each block and the
 * elements are assigned to processors within each block by greedily filling
 * each available processors' allowance by contiguous intervals along the
 * Z-order curve. Some examples:
 * - If there are 8 blocks, 16 elements per block, 16 cores, and all cores are
 * allowed to have elements: each core gets an allowance of 128 / 16 = 8
 * elements, so each core gets half of a block, and the 8 elements for each core
 * within the block are chosen via Z-order curve for the respective blocks.
 * - If there are 3 blocks, 4 elements per block, 4 cores, and all cores are
 * allowed to have elements: each core gets an allowance of 12 / 4 = 3 elements.
 * Core 0 gets three elements from the first block, core 1 gets one element from
 * the first block and two elements from the second block, core 2 gets two
 * elements from the second block and one from the third, and core 3 gets the
 * remaining three elements from the third block. Each collection of elements
 * within the blocks are then assigned using intervals along the Z-order curve
 * for each block.
 * - Same as the previous example, 3 blocks, 4 elements per block, and 4 cores,
 * except now we require that physical cores 1 and 3 don't have any elements on
 * them. The new distribution would look like:
 *   - Elements on old core 0 -> new core 0
 *   - No elements on new core 1
 *   - Elements on old core 1 -> new core 2
 *   - No elements on new core 3
 *   - Elements on old core 2 -> new core 4
 *   - Elements on old core 3 -> new core 5
 *
 * \note In the third example, even though only 4 cores are used to place
 * elements, the simulation is required to be run on at least 6 cores (4 cores
 * for elements + 2 cores without elements)
 *
 * Morton curves are a simple and easily-computed space-filling curve that
 * (unlike Hilbert curves) permit diagonal traversal. See, for instance,
 * \cite Borrell2018 for a discussion of mesh partitioning using space-filling
 * curves.
 * A concrete example of the use of a Morton curve in 2d is given below.
 *
 * A sketch of a 2D block with 4x2 elements, with each element labeled according
 * to the order on the Morton curve:
 * ```
 *          x-->
 *          0   1   2   3
 *        ----------------
 *  y  0 |  0   2   4   6
 *  |    |  | / | / | / |
 *  v  1 |  1   3   5   7
 * ```
 * (forming a zig-zag path, that under some rotation/reflection has a 'Z'
 * shape).
 *
 * The Morton curve method is a quick way of getting acceptable spatial locality
 * -- usually, for approximately even distributions, it will ensure that
 * elements are assigned in large volume chunks, and the structure of the Morton
 * curve ensures that for a given processor and block, the elements will be
 * assigned in no more than two orthogonally connected clusters. In principle, a
 * Hilbert curve could potentially improve upon the gains obtained by this class
 * by guaranteeing that all elements within each block form a single
 * orthogonally connected cluster.
 *
 * The assignment of portions of blocks to processors may use partial blocks,
 * and/or multiple blocks to ensure an even distribution of elements to
 * processors.
 * We currently make no distinction between dividing elements between processors
 * within a node and dividing elements between processors across nodes. The
 * current technique aims to have a simple method of reducing communication
 * globally, though it would likely be more efficient to prioritize minimization
 * of inter-node communication, because communication across interconnects is
 * the primary cost of communication in charm++ runs.
 *
 * \warning The use of the Morton curve to generate a well-clustered element
 * distribution currently assumes that the refinement is uniform over each
 * block, with no internal structure that would be generated by, for instance
 * AMR.
 * This distribution method will need alteration to perform well for blocks with
 * internal structure from h-refinement. Morton curves can be defined
 * recursively, so a generalization of the present method is possible for blocks
 * with internal refinement
 */
struct WeightedBlockZCurveProcDistribution {
  /// The `number_of_procs_with_elements` argument represents how many procs
  /// will have elements. This is not necessarily equal to the total number of
  /// procs because some global procs may be ignored by the third argument
  /// `global_procs_to_ignore`
  // WeightedBlockZCurveProcDistribution(
  //     size_t number_of_procs_with_elements,
  //     const std::vector<std::vector<double>>& cost_by_element_by_block,
  //     const std::unordered_set<size_t>& global_procs_to_ignore = {});

  WeightedBlockZCurveProcDistribution(
    const size_t number_of_procs_with_elements,
    // TODO : this needs to be traversed according to morton curve?
    const std::vector<std::vector<double> >& cost_by_element_by_block,
    const std::unordered_set<size_t>& global_procs_to_ignore = {}) {
  // initialize distribution
  block_element_distribution_ =
      std::vector<std::vector<std::pair<size_t, size_t> > >(
          cost_by_element_by_block.size());

  // size_t total_elements = 0;
  double total_cost = 0.0;
  // double min_cost = std::numeric_limits<double>::max();
  // double max_cost = std::numeric_limits<double>::min();

  for (auto& block : cost_by_element_by_block) {
    // const auto min_element_this_block = alg::min_element(block);
    // const auto max_element_this_block = alg::max_element(block);
    // if (min_element_this_block < min_cost) {
    //   min_cost = min_element_this_block;
    // }
    // if (max_element_this_block > max_cost) {
    //   max_cost = max_element_this_block;
    // }

    // total_elements += block.size();
    for (double element_cost : block) {
      total_cost += element_cost;
    }
  }

  // std::cout << "total_cost : " << total_cost << std::endl;

  // const double cost_range = max_cost - min_cost;

  // const size_t num_elements_per_proc =
  //     number_of_elements / number_of_procs_with_elements;
  // const size_t num_elements_remainder =
  //     number_of_elements -
  //     (num_elements_per_proc * number_of_procs_with_elements);

  // TODO : need to address the case where one single
  // element cost is much greater than the cost_allowance_per_proc,
  // e.g. maybe that would mean taking p-refinement into account

  const double cost_allowance_per_proc =
      total_cost / number_of_procs_with_elements;

  // std::cout << "cost_allowance_per_proc : " << cost_allowance_per_proc
  //           << std::endl;

  // size_t remaining_elements_in_block = cost_by_element_by_block[0].size();
  size_t current_block = 0;
  size_t current_element_of_current_block = 0;
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
    // std::cout << "global_proc_number : " << global_proc_number << std::endl;

    // initialize cost for this proc to be the current element
    // double cost_spent_on_proc =
    //     cost_by_element_by_block[current_block][current_element_of_current_block];
    // size_t num_elements_distributed_to_proc = 1;
    double cost_spent_on_proc = 0.0;
    // size_t num_elements_distributed_to_proc = 0;
    // while we still have cost allowed on the proc
    while (current_block < cost_by_element_by_block.size() and
           cost_spent_on_proc <= cost_allowance_per_proc) {
      // std::cout << "current_block : " << current_block << std::endl;
      // std::cout << "current_element_of_current_block : "
      //           << current_element_of_current_block << std::endl;
      const size_t num_elements_current_block =
          cost_by_element_by_block[current_block].size();
      // while we still have elements left on the block and we still
      // have cost allowed on the proc
      size_t num_elements_distributed_to_proc = 0;
      // std::cout << "begin while : " << std::endl;
      // std::cout << "cost_spent_on_proc before : " << cost_spent_on_proc
      //           << std::endl;
      while (current_element_of_current_block < num_elements_current_block and
             cost_spent_on_proc <= cost_allowance_per_proc) {
        cost_spent_on_proc +=
            cost_by_element_by_block[current_block]
                                    [current_element_of_current_block];
        num_elements_distributed_to_proc++;
        current_element_of_current_block++;
      }
      // std::cout << "end while : " << std::endl;
      // std::cout << "cost_spent_on_proc after : " << cost_spent_on_proc
      //           << std::endl;

      block_element_distribution_.at(current_block)
          .emplace_back(std::make_pair(global_proc_number,
                                       num_elements_distributed_to_proc));
      if (current_element_of_current_block >=
          cost_by_element_by_block[current_block].size()) {
        // whole block has been distributed
        ++current_block;
        current_element_of_current_block = 0;
      }
    }
  }
}

  /// Gets the suggested processor number for a particular element,
  /// determined by the greedy block assignment and Morton curve element
  /// assignment described in detail in the parent class documentation.
  template <size_t Dim>
size_t get_proc_for_element(
    const ElementId<Dim>& element_id) const {
  // the index of the element on the z-curve?
  const size_t element_order_index = z_curve_index(element_id);
  size_t total_so_far = 0;
  // iterating over arbitrary # of procs for some block element_id is in? where
  // element_info is an element allowance for a specific processor:
  //     (proc #, # of allowed elements)
  for (const std::pair<size_t, size_t>& element_info :
       gsl::at(block_element_distribution_, element_id.block_id())) {
    // if total allowance so far <= z order index of element and
    // (this processor's # of allowed elements + total allowance so far) >
    // this element's z order index
    //     return this processor
    if (total_so_far <= element_order_index and
        element_info.second + total_so_far > element_order_index) {
      return element_info.first;
    }
    // otherwise, add this processor's allowance to the total so far & continue
    total_so_far += element_info.second;
  }
  ERROR(
      "Processor not successfully chosen. This indicates a flaw in the logic "
      "of WeightedBlockZCurveProcDistribution.");
}


  std::vector<std::vector<std::pair<size_t, size_t>>>
  block_element_distribution() const {
    return block_element_distribution_;
  }

 private:
  // in this nested data structure:
  // - The block id is the first index
  // - There is an arbitrary number of CPUs per block, each with an element
  //   allowance
  // - Each element allowance is represented by a pair of proc number, number of
  //   elements in the allowance
  //   const size_t number_of_procs_with_elements_;
  //   const std::unordered_set<size_t>& global_procs_to_ignore_;
  std::vector<std::vector<std::pair<size_t, size_t>>>
      block_element_distribution_;
};
}  // namespace domain
