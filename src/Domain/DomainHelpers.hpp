// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DomainHelper functions

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include "Utilities/ConstantExpressions.hpp"

/// \cond
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
template <size_t VolumeDim>
class BlockNeighbor;
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
template <size_t VolumeDim>
class Direction;
namespace Frame {
struct Logical;
}  // namespace Frame
/// \endcond

/// \ingroup ComputationalDomainGroup
/// Each member in `PairOfFaces` holds the global corner ids of a block face.
/// `PairOfFaces` is used in setting up periodic boundary conditions by
/// identifying the two faces with each other.
/// \requires The pair of faces must belong to a single block.
struct PairOfFaces {
  std::vector<size_t> first;
  std::vector<size_t> second;
};

/// \ingroup ComputationalDomainGroup
/// Sets up the BlockNeighbors using the corner numbering scheme
/// to deduce the correct neighbors and orientations. Does not set
/// up periodic boundary conditions.
template <size_t VolumeDim>
void set_internal_boundaries(
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<std::vector<
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks);

/// \ingroup ComputationalDomainGroup
/// Sets up additional BlockNeighbors corresponding to any
/// periodic boundary condtions provided by the user. These are
/// stored in identifications.
template <size_t VolumeDim>
void set_periodic_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<std::vector<
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks);

/// \ingroup ComputationalDomainGroup
/// These are the CoordinateMaps of the Wedge3Ds used in the Sphere, Shell, and
/// binary compact object DomainCreators. This function can also be used to
/// wrap the Sphere or Shell in a cube made of six Wedge3Ds.
/// The argument `x_coord_of_shell_center` specifies a translation of the Shell
/// in the x-direction in the TargetFrame. For example, the BBH DomainCreator
/// uses this to set the position of each BH.
/// When the argument `use_half_wedges` is set to `true`, the wedges in the
/// +z,-z,+y,-y directions are cut in half along their xi-axes. The resulting
/// ten CoordinateMaps are used for the outermost Blocks of the BBH Domain.
template <typename TargetFrame>
std::vector<std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
wedge_coordinate_maps(double inner_radius, double outer_radius,
                      double inner_sphericity, double outer_sphericity,
                      bool use_equiangular_map,
                      double x_coord_of_shell_center = 0.0,
                      bool use_half_wedges = false) noexcept;

/// \ingroup ComputationalDomainGroup
/// These are the ten Frustums used in the DomainCreators for binary compact
/// objects. The cubes enveloping the two Shells each have a side length of
/// `length_inner_cube`. The ten frustums also make up a cube of their own,
/// of side length `length_outer_cube`.
template <typename TargetFrame>
std::vector<std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
frustum_coordinate_maps(double length_inner_cube, double length_outer_cube,
                        bool use_equiangular_map) noexcept;
