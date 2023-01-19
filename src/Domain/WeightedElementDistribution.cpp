// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/WeightedElementDistribution.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain {

namespace {
// This interleaves the bits of the element index.
// A sketch of a 2D block with 4x2 elements, with bit indices and resulting
// z-curve
//
//        x-->
//        00  01  10  11
// y  0 |  0   2   4   6
// |    |
// v  1 |  1   3   5   7
template <size_t Dim>
size_t z_curve_index(const ElementId<Dim>& element_id) {
  // for the bit manipulation of the element index, we need to interleave the
  // indices in each dimension in order according to how many bits are in the
  // index representation. This variable stores the refinement level and
  // dimension index in ascending order of refinement level, representing a
  // permutation of the dimensions
  // pair<refinement level, dim index> in order of ascending refinement
  std::array<std::pair<size_t, size_t>, Dim>
      dimension_by_highest_refinement_level;
  for (size_t i = 0; i < Dim; ++i) {
    dimension_by_highest_refinement_level.at(i) =
        std::make_pair(element_id.segment_id(i).refinement_level(), i);
  }
  alg::sort(dimension_by_highest_refinement_level,
            [](const std::pair<size_t, size_t>& lhs,
               const std::pair<size_t, size_t>& rhs) {
              return lhs.first < rhs.first;
            });

  size_t element_order_index = 0;

  // 'gap' the lowest refinement direction bits as:
  // ... x1 x0 -> ... x1 0 0 x0,
  // then bitwise or in 'gap'ed and shifted next-lowest refinement direction
  // bits as:
  // ... y2 y1 y0 -> ... y2 0 y1 x1 0 y0 x0
  // then bitwise or in 'gap'ed and shifted highest-refinement direction bits
  // as:
  // ... z3 z2 z1 z0 -> z3 z2 y2 z1 y1 x1 z0 y0 x0
  // note that we must skip refinement-level 0 dimensions as though they are
  // not present
  size_t leading_gap = 0;
  for (size_t i = 0; i < Dim; ++i) {
    const size_t id_to_gap_and_shift =
        element_id
            .segment_id(
                gsl::at(dimension_by_highest_refinement_level, i).second)
            .index();
    size_t total_gap = leading_gap;
    if (gsl::at(dimension_by_highest_refinement_level, i).first > 0) {
      ++leading_gap;
    }
    for (size_t bit_index = 0;
         bit_index < gsl::at(dimension_by_highest_refinement_level, i).first;
         ++bit_index) {
      // This operation will not overflow for our present use of `ElementId`s.
      // This technique densely assigns an ElementID a unique size_t identifier
      // determining the Morton curve order, and `ElementId` supports refinement
      // levels such that a global index within a block will fit in a 64-bit
      // unsigned integer.
      element_order_index |=
          ((id_to_gap_and_shift & two_to_the(bit_index)) << total_gap);
      for (size_t j = 0; j < Dim; ++j) {
        if (i != j and
            bit_index + 1 <
                gsl::at(dimension_by_highest_refinement_level, j).first) {
          ++total_gap;
        }
      }
    }
  }
  return element_order_index;
}

template <size_t Dim>
std::array<size_t, Dim> element_id_from_z_curve_index(
    const size_t z_order_index,
    const std::array<size_t, Dim>& block_refinements) {
  std::array<std::pair<size_t, size_t>, Dim>
      dimension_by_highest_refinement_level;
  for (size_t i = 0; i < Dim; ++i) {
    dimension_by_highest_refinement_level.at(i) =
        std::make_pair(gsl::at(block_refinements, i), i);
  }
  // {{L, Dim}, {L, Dim}, ...}
  alg::sort(dimension_by_highest_refinement_level,
            [](const std::pair<size_t, size_t>& lhs,
               const std::pair<size_t, size_t>& rhs) {
              return lhs.first < rhs.first;
            });
  
  std::array<std::pair<size_t, size_t>, Dim>
      segment_indices_by_highest_refinement_level;
  for (size_t i = 0; i < Dim; ++i) {
    segment_indices_by_highest_refinement_level.at(i) =
        std::make_pair(0, gsl::at(dimension_by_highest_refinement_level, i).second);
  }
  size_t starting_dim_index = 0;
  size_t bit_index = 0;
  size_t element_order_index = z_order_index;
      
  // TODO : try to optimize by handling last case separately when it's the
  // remaining bits of the highest refinement, i.e. probably do:
  // starting_dim_index < (Dim - 1) then handle the last case separately after
  while (starting_dim_index < Dim) {
    const size_t refinement_level =
      gsl::at(dimension_by_highest_refinement_level, starting_dim_index).first;
    // TODO : this is probably wrong
    // const size_t num_refinement_bits =
    //     bit_index == 0 ? two_to_the(refinement_level) : 
    //         two_to_the(refinement_level) - two_to_the(bit_index);
    const size_t num_refinement_bits =
        refinement_level - bit_index;
    for (size_t i = 0; i < num_refinement_bits; i++) {
      for (size_t dim_index = starting_dim_index; dim_index < Dim; dim_index++) {
        segment_indices_by_highest_refinement_level.at(dim_index).first |=
            ((element_order_index & 1) << bit_index);
        element_order_index >>= 1;
      }
      bit_index++;
    }

    starting_dim_index++;
    while (starting_dim_index < Dim and
           refinement_level ==
               gsl::at(dimension_by_highest_refinement_level, starting_dim_index).first) {
      starting_dim_index++;
    }
  }

  std::array<size_t, Dim> element_id{};

  for (size_t i = 0; i < Dim; i++) {
    gsl::at(element_id,
       gsl::at(segment_indices_by_highest_refinement_level, i).second) =
           gsl::at(segment_indices_by_highest_refinement_level, i).first;
  }
  
  return element_id;
}
}  // namespace

WeightedBlockZCurveProcDistribution::WeightedBlockZCurveProcDistribution(
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
size_t WeightedBlockZCurveProcDistribution::get_proc_for_element(
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
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                               \
  template size_t WeightedBlockZCurveProcDistribution::get_proc_for_element( \
      const ElementId<GET_DIM(data)>& element_id) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace domain
