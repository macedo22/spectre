// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

template <size_t Dim>
class ElementId;

namespace domain {

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
size_t z_curve_index_from_element_id(const ElementId<Dim>& element_id);

template <size_t Dim>
std::array<size_t, Dim> segment_indices_from_z_curve_index(
    const size_t z_order_index,
    const std::array<size_t, Dim>& block_refinements);

template <size_t Dim>
std::vector<ElementId<Dim>> initial_element_ids_in_z_curve_order(
    size_t block_id, std::array<size_t, Dim> initial_ref_levs,
    size_t grid_index = 0);

}  // namespace domain
