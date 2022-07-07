// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/WeightedElementDistribution.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
// #include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Domain/ZCurveIndex.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain {
template <size_t Dim>
WeightedBlockZCurveProcDistribution<Dim>::WeightedBlockZCurveProcDistribution(
    const size_t number_of_procs_with_elements,
    // TODO : this needs to be traversed according to morton curve?
    const std::vector<std::vector<double> >& cost_by_element_by_block,
    const std::unordered_set<size_t>& global_procs_to_ignore) {
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

// just a getter? jumps through z curve until it finds the right section of it
// and thus the proc number?
template <size_t Dim>
size_t WeightedBlockZCurveProcDistribution<Dim>::get_proc_for_element(
    const ElementId<Dim>& element_id) const {
  // the index of the element on the z-curve?
  const size_t element_order_index = z_curve_index_from_element_id(element_id);
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

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class WeightedBlockZCurveProcDistribution<GET_DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace domain
