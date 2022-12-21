// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementDistribution.hpp"

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

// namespace {
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
  // std::cout << "dimension_by_highest_refinement_level {L, Dim} : {{"
  //   << dimension_by_highest_refinement_level[0].first << ", "
  //   << dimension_by_highest_refinement_level[0].second << "}";

  // for (size_t i = 1; i < Dim; i++) {
  //   std::cout << ", {"
  //   << dimension_by_highest_refinement_level[i].first << ", "
  //   << dimension_by_highest_refinement_level[i].second << "}";
  // }

  // std::cout << "}" << std::endl;

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
    // std::cout << "Dim index (i) : " << i << std::endl;
    // std::cout << "Dim at i : " << gsl::at(dimension_by_highest_refinement_level, i).second << std::endl;
    // std::cout << "element_id.segment_id(that) : " << element_id
    //         .segment_id(
    //             gsl::at(dimension_by_highest_refinement_level, i).second) << std::endl;
    // std::cout << "element_id.segment_id(...).index() : " << element_id
    //         .segment_id(
    //             gsl::at(dimension_by_highest_refinement_level, i).second)
    //         .index() << std::endl;

    const size_t id_to_gap_and_shift =
        element_id
            .segment_id(
                gsl::at(dimension_by_highest_refinement_level, i).second)
            .index();
    // std::cout << "id_to_gap_and_shift : " << id_to_gap_and_shift << std::endl;
    size_t total_gap = leading_gap;
    if (gsl::at(dimension_by_highest_refinement_level, i).first > 0) {
      ++leading_gap;
    }
    // std::cout << "leading_gap : " << leading_gap << std::endl;
    for (size_t bit_index = 0;
         // while less than # of refinement levels
         bit_index < gsl::at(dimension_by_highest_refinement_level, i).first;
         ++bit_index) {
      // std::cout << "bit_index (for ref. level indexing) : " << bit_index std::endl;
      // This operation will not overflow for our present use of `ElementId`s.
      // This technique densely assigns an ElementID a unique size_t identifier
      // determining the Morton curve order, and `ElementId` supports refinement
      // levels such that a global index within a block will fit in a 64-bit
      // unsigned integer.
      // std::cout << "element_order_index != "
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

  // std::cout << "element_order_index : " << element_order_index << std::endl;
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
  
  std::cout << "dimension_by_highest_refinement_level {L, Dim} : {{"
    << dimension_by_highest_refinement_level[0].first << ", "
    << dimension_by_highest_refinement_level[0].second << "}";

  for (size_t i = 1; i < Dim; i++) {
    std::cout << ", {"
    << dimension_by_highest_refinement_level[i].first << ", "
    << dimension_by_highest_refinement_level[i].second << "}";
  }

  std::cout << "}" << std::endl;
  
  // pairs are: {SegmentId index, dim}, where dim = 0, 1, or 2 for x, y, or z
  // init as {{0, least refined dim}, {0, next least}, ...}
  std::array<std::pair<size_t, size_t>, Dim>
      segment_indices_by_highest_refinement_level;
  for (size_t i = 0; i < Dim; ++i) {
    segment_indices_by_highest_refinement_level.at(i) =
        std::make_pair(0, gsl::at(dimension_by_highest_refinement_level, i).second);
  }

  std::cout << "BEFORE segment_indices_by_highest_refinement_level {Index, Dim} : {{"
    << segment_indices_by_highest_refinement_level[0].first << ", "
    << segment_indices_by_highest_refinement_level[0].second << "}";

  for (size_t i = 1; i < Dim; i++) {
    std::cout << ", {"
    << segment_indices_by_highest_refinement_level[i].first << ", "
    << segment_indices_by_highest_refinement_level[i].second << "}";
  }

  std::cout << "}" << std::endl;

  // const highest_refinement_level =
  //     gsl::at(dimension_by_highest_refinement_level, Dim - 1);
  // // initialize to lowest refinement
  // size_t refinement_level =
  //     gsl::at(dimension_by_highest_refinement_level, 0);
  // // size_t refinements_processed = 0;
  // const size_t num_refinement_bits = two_to_the(refinement_level);
  size_t starting_dim_index = 0;
  // size_t mask_index = 0;
  size_t bit_index = 0;
  size_t element_order_index = z_order_index;
  std::cout << "z_order_index : " << z_order_index << std::endl;
      
  // size_t refinement_levels_processed = 0;
  // TODO : try to optimize by handling last case separately when it's the
  // remaining bits of the highest refinement, i.e. probably do:
  // starting_dim_index < (Dim - 1) then handle the last case separately after
  while (starting_dim_index < Dim) {
    std::cout << "starting_dim_index : " << starting_dim_index << std::endl;
    const size_t refinement_level =
      gsl::at(dimension_by_highest_refinement_level, starting_dim_index).first;
    // TODO : this is probably wrong
    // const size_t num_refinement_bits =
    //     bit_index == 0 ? two_to_the(refinement_level) : 
    //         two_to_the(refinement_level) - two_to_the(bit_index);
    const size_t num_refinement_bits =
        refinement_level - bit_index;
    std::cout << "refinement_level : " << refinement_level << std::endl;
    std::cout << "num_refinement_bits : " << num_refinement_bits << std::endl;
    for (size_t i = 0; i < num_refinement_bits; i++) {
      std::cout << "i : " << i << std::endl;
      // const size_t shift =
      //     bit_index == 0 ? 0 : 
      //       bit_index - 1;
      for (size_t dim_index = starting_dim_index; dim_index < Dim; dim_index++) {
        std::cout << "dim_index : " << dim_index << std::endl;
        // segment_indices_by_highest_refinement_level.at(dim_index).first |=
        //     ((z_order_index & (two_to_the(mask_index))) >> (mask_index - bit_index));
        // // mask << 1;
        // mask_index++;
        std::cout << "current segment index BEFORE |= : "
                  << segment_indices_by_highest_refinement_level.at(dim_index).first << std::endl;
        segment_indices_by_highest_refinement_level.at(dim_index).first |=
            ((element_order_index & 1) << bit_index);
        std::cout << "current segment index AFTER |= : "
                  << segment_indices_by_highest_refinement_level.at(dim_index).first << std::endl;
        element_order_index >>= 1;
        std::cout << "element_order_index updated to : " << element_order_index << std::endl;
      }
      bit_index++;
      std::cout << "bit_index updated to : " << bit_index << std::endl;
    }

    // mask_index += num_refinement_bits * (dim_index - starting_dim_index);
    // bit_index = two_to_the(refinement_level);
    // bit_index += num_refinement_bits;
    // // bit_index += (num_refinement_bits * (dim_index - starting_dim_index));
    // refinement_levels_processed = refinement_level;

    starting_dim_index++;
    while (starting_dim_index < Dim and
           refinement_level ==
               gsl::at(dimension_by_highest_refinement_level, starting_dim_index).first) {
      // starting_dim_index++;
      // refinement_level =
      //     gsl::at(dimension_by_highest_refinement_level, starting_dim_index).first;
      starting_dim_index++;
    }
    std::cout << "starting_dim_index at end of loop : " << starting_dim_index << std::endl;
  }

  std::cout << "AFTER segment_indices_by_highest_refinement_level {Index, Dim} : {{"
    << segment_indices_by_highest_refinement_level[0].first << ", "
    << segment_indices_by_highest_refinement_level[0].second << "}";

  for (size_t i = 1; i < Dim; i++) {
    std::cout << ", {"
    << segment_indices_by_highest_refinement_level[i].first << ", "
    << segment_indices_by_highest_refinement_level[i].second << "}";
  }

  std::cout << "}" << std::endl;

  // alg::sort(segment_indices_by_highest_refinement_level,
  //           [](const std::pair<size_t, size_t>& lhs,
  //              const std::pair<size_t, size_t>& rhs) {
  //             return lhs.second < rhs.second;
  //           });
  std::array<size_t, Dim> element_id{};

  for (size_t i = 0; i < Dim; i++) {
    gsl::at(element_id,
       gsl::at(segment_indices_by_highest_refinement_level, i).second) =
           gsl::at(segment_indices_by_highest_refinement_level, i).first;
  }
  
  return element_id;
  

  // // const highest_refinement_level =
  // //     gsl::at(dimension_by_highest_refinement_level, Dim - 1);
  // // // initialize to lowest refinement
  // // size_t refinement_level =
  // //     gsl::at(dimension_by_highest_refinement_level, 0);
  // // // size_t refinements_processed = 0;
  // // const size_t num_refinement_bits = two_to_the(refinement_level);
  // size_t starting_dim_index = 0;
  // size_t mask_index = 0;
  // size_t bit_index = 0;
  // // size_t refinement_levels_processed = 0;
  // while (starting_dim_index < Dim) {
  //   const size_t refinement_level =
  //     gsl::at(dimension_by_highest_refinement_level, starting_dim_index).first;
  //   // TODO : this is probably wrong
  //   const size_t num_refinement_bits =
  //       bit_index == 0 ? two_to_the(refinement_level) : 
  //           two_to_the(refinement_level) - two_to_the(bit_index);
  //   for (size_t i = 0; i < num_refinement_bits; i++) {
  //     // const size_t shift =
  //     //     bit_index == 0 ? 0 : 
  //     //       bit_index - 1;
  //     for (size_t dim_index = starting_dim_index; dim_index < Dim - 1; dim_index++) {
  //       segment_indices_by_highest_refinement_level.at(dim_index).first |=
  //           ((z_order_index & (two_to_the(mask_index))) >> (mask_index - bit_index));
  //       // mask << 1;
  //       mask_index++;
  //     }
  //     bit_index++;
  //   }

  //   // mask_index += num_refinement_bits * (dim_index - starting_dim_index);
  //   // bit_index = two_to_the(refinement_level);
  //   // bit_index += num_refinement_bits;
  //   // // bit_index += (num_refinement_bits * (dim_index - starting_dim_index));
  //   // refinement_levels_processed = refinement_level;

  //   starting_dim_index++;
  //   while (starting_dim_index < Dim and
  //          refinement_level ==
  //              gsl::at(dimension_by_highest_refinement_level, starting_dim_index).first) {
  //     // starting_dim_index++;
  //     // refinement_level =
  //     //     gsl::at(dimension_by_highest_refinement_level, starting_dim_index).first;
  //     starting_dim_index++;
  //   }

  //   // while (starting_dim_index < (Dim - 1) and
  //   //        refinement_level ==
  //   //            gsl::at(dimension_by_highest_refinement_level, starting_dim_index).first) {
  //   //   starting_dim_index++;
  //   //   refinement_level =
  //   //       gsl::at(dimension_by_highest_refinement_level, starting_dim_index).first;
  //   //   starting_dim_index++;
  //   // }
  // }

  // const highest_refinement_level =
  //     gsl::at(dimension_by_highest_refinement_level, Dim - 1);
  // // initialize to lowest refinement
  // size_t refinement_level =
  //     gsl::at(dimension_by_highest_refinement_level, 0);
  // // size_t refinements_processed = 0;
  // const size_t num_refinement_bits = two_to_the(refinement_level);
  // size_t starting_dim_index = 0;
  // size_t mask = 1;
  // while (true) {
  //   for (size_t i = 0; i < num_refinement_bits; i++) {
  //     for (size_t dim_index = starting_dim_index; dim_index < Dim - 1; dim_index++) {
  //       segment_indices_by_highest_refinement_level.at(dim_index).first +=
  //           z_order_index & mask;
  //       mask << 1;
  //     }
  //   }

  //   if (refinement_level == highest_refinement_level) {
  //     break; 
  //   }

  //   starting_dim_index++;
  //   refinement_level = dimension_by_highest_refinement_level.at(starting_dim_index);
  //   num_refinement_bits = two_to_the(refinement_level);

  //   // refinement_level =
  //   //     dimension_by_highest_refinement_level.at(starting_dim_index + 1).second;
    
  //   // size_t next_refinement_level = refinement_level;
  //   // while (refinement_level < highest_refinement_level
  //   //        and next_refinement_level == refinement_level) {
      
  //   // }
  // }

  // // TODO : handle when refinement = 0?;
  // size_t refinement_levels_left =
  //     gsl::at(dimension_by_highest_refinement_level); 
  // for (size_t i = 0; i < Dim; i++) {
  //   const size_t refinement =
  //       gsl::at(dimension_by_highest_refinement_level, i);
  //   // TODO : handle case with 0 refinement?
  //   const size_t num_refinement_bits = two_to_the(refinement);

  //   size_t segment_id_index = 0;
  //   size_t mask = 1;
  //   for (size_t bit_index = 0; bit_index < num_refinement_bits * (Dim - i); bit_index++) {
  //     segment_id_index += z_order_index & mask;
  //     mask *= 2;
  //   }

  // }
}
// }  // namespace

template <size_t Dim>
BlockZCurveProcDistribution<Dim>::BlockZCurveProcDistribution(
    size_t number_of_procs_with_elements,
    const std::vector<std::array<size_t, Dim>>& refinements_by_block,
    const std::unordered_set<size_t>& global_procs_to_ignore) {
  // initialize distribution
  block_element_distribution_ =
      std::vector<std::vector<std::pair<size_t, size_t>>>(
          refinements_by_block.size());
  auto add_number_of_elements_for_refinement =
      [](size_t lhs, const std::array<size_t, Dim>& rhs) {
        size_t value = 1;
        for (size_t i = 0; i < Dim; ++i) {
          // value *= 2^[refinement along one dimension]
          value *= two_to_the(gsl::at(rhs, i));
        }
        return lhs + value;
      };
  // total number of elements
  const size_t number_of_elements =
      std::accumulate(refinements_by_block.begin(), refinements_by_block.end(),
                      0_st, add_number_of_elements_for_refinement);
  ASSERT(not refinements_by_block.empty(),
         "`refinements_by_block` must be non-empty.");
  // currently, we just assign uniform weight to elements. In future, it will
  // probably be better to take into account p-refinement, but then the z-curve
  // method will also require weighting.
  // first block's
  size_t remaining_elements_in_block =
      add_number_of_elements_for_refinement(0_st, refinements_by_block[0]);
  size_t current_block = 0;
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
    // remaining_elements_on_proc = num elements / num procs
    // + 1 if current proc index (i) is a remainder
    size_t remaining_elements_on_proc =
        (number_of_elements / number_of_procs_with_elements) +
        (i < (number_of_elements % number_of_procs_with_elements) ? 1 : 0);
    // while we still have elements to fill on the proc
    while (remaining_elements_on_proc > 0) {
      // for this block's list of proc allowances, append
      // (current proc,
      //  min(remaining elements in block, remaining elements on proc))
      block_element_distribution_.at(current_block)
          .emplace_back(std::make_pair(global_proc_number,
                                       std::min(remaining_elements_in_block,
                                                remaining_elements_on_proc)));
      if (remaining_elements_in_block <= remaining_elements_on_proc) {
        // if we just assigned remaining elements in block, then
        // subtract that number from the remaining elements on the proc
        remaining_elements_on_proc -= remaining_elements_in_block;
        // increment to next block
        ++current_block;
        // if this next block (current_block) index is in range of all blocks
        if (current_block < refinements_by_block.size()) {
          // remaining_elements_in_block = this next block's number of elements,
          // repeat loop
          remaining_elements_in_block = add_number_of_elements_for_refinement(
              0_st, gsl::at(refinements_by_block, current_block));
        }
      } else {
        // if we just assigned remainnig elements on proc, then
        // subtract that number from the remaining elements in the block and
        // we have no more remaining element allowance on this processor
        remaining_elements_in_block -= remaining_elements_on_proc;
        remaining_elements_on_proc = 0;
      }
    }
  }
}

// just a getter? jumps through z curve until it finds the right section of it
// and thus the proc number?
template <size_t Dim>
size_t BlockZCurveProcDistribution<Dim>::get_proc_for_element(
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
      "of BlockZCurveProcDistribution.");
}
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class BlockZCurveProcDistribution<GET_DIM(data)>; \
  template size_t z_curve_index(const ElementId<GET_DIM(data)>& element_id); \
  template std::array<size_t, GET_DIM(data)> element_id_from_z_curve_index( \
    const size_t z_order_index, \
    const std::array<size_t, GET_DIM(data)>& block_refinements);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace domain
