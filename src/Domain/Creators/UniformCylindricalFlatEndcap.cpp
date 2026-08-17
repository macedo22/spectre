// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/UniformCylindricalFlatEndcap.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcap.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/ParseError.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {
UniformCylindricalFlatEndcap::UniformCylindricalFlatEndcap(
    const typename SphereCenter::type sphere_center,
    const typename SphereRadius::type sphere_radius,
    const typename CylinderCenter::type cylinder_center,
    const typename CylinderRadius::type cylinder_radius,
    const typename ZPlane::type z_plane,
    const typename InitialRadialGridPoints::type initial_radial_grid_points,
    const typename InitialThetaGridPoints::type initial_theta_grid_points,
    const typename InitialZGridPoints::type initial_z_grid_points,
    const typename InitialRefinementInZ::type initial_refinement_in_z,
    const Options::Context& context)
    : sphere_center_(sphere_center),
      sphere_radius_(sphere_radius),
      cylinder_center_(cylinder_center),
      cylinder_radius_(cylinder_radius),
      z_plane_(z_plane),
      initial_radial_grid_points_(initial_radial_grid_points),
      initial_theta_grid_points_(initial_theta_grid_points),
      initial_z_grid_points_(initial_z_grid_points),
      initial_refinement_in_z_(initial_refinement_in_z) {
  if (sphere_radius_ <= 0.0) {
    PARSE_ERROR(context,
                "SphereRadius must be positive, but is: " << sphere_radius_);
  }

  if (cylinder_radius_ <= 0.0) {
    PARSE_ERROR(context, "CylinderRadius must be positive, but is: "
                             << cylinder_radius_);
  }

  if (z_plane_ <= sphere_center_[2]) {
    PARSE_ERROR(context,
                "ZPlane must be > z coordinate of SphereCenter, but ZPlane is "
                    << z_plane_ << " and SphereCenter z coordinate is "
                    << sphere_center_[2]);
  }
  if (z_plane_ >= sphere_center_[2] + sphere_radius_) {
    PARSE_ERROR(context,
                "ZPlane must be < (SphereRadius + z coordinate of "
                "SphereCenter), but ZPlane is "
                    << z_plane_
                    << " and (SphereRadius + z coordinate of SphereCenter) is "
                    << (sphere_center_[2] + sphere_radius_));
  }
  if (z_plane_ >= cylinder_center_[2]) {
    PARSE_ERROR(
        context,
        "ZPlane must be < z coordinate of CylinderCenter, but ZPlane is "
            << z_plane_ << " and CylinderCenter z coordinate is "
            << cylinder_center_[2]);
  }

  if (initial_theta_grid_points_ % 2 != 1) {
    PARSE_ERROR(context,
                "The number of angular grid points must be odd (this helps "
                "with numerical stability), but got "
                    << initial_theta_grid_points_);
  }
  if (initial_theta_grid_points_ > 4 * initial_radial_grid_points_ - 3) {
    PARSE_ERROR(context,
                "The number of angular grid points must be <= 4 * "
                "(the number of radial grid points) - 3, but got n_r = "
                    << initial_radial_grid_points_
                    << "and n_theta = " << initial_theta_grid_points_);
  }
}

UniformCylindricalFlatEndcap::UniformCylindricalFlatEndcap(
    const typename SphereCenter::type sphere_center,
    const typename SphereRadius::type sphere_radius,
    const typename CylinderCenter::type cylinder_center,
    const typename CylinderRadius::type cylinder_radius,
    const typename ZPlane::type z_plane,
    const typename InitialThetaGridPoints::type initial_radial_grid_points,
    const typename InitialThetaGridPoints::type initial_theta_grid_points,
    const typename InitialZGridPoints::type initial_z_grid_points,
    const typename InitialRefinementInZ::type initial_refinement_in_z,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        lower_z_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        upper_z_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        mantle_boundary_condition,
    const Options::Context& context)
    : UniformCylindricalFlatEndcap(
          sphere_center, sphere_radius, cylinder_center, cylinder_radius,
          z_plane, initial_radial_grid_points, initial_theta_grid_points,
          initial_z_grid_points, initial_refinement_in_z, context) {
  // NOLINTNEXTLINE
  lower_z_boundary_condition_ = std::move(lower_z_boundary_condition);
  // NOLINTNEXTLINE
  upper_z_boundary_condition_ = std::move(upper_z_boundary_condition);
  // NOLINTNEXTLINE
  mantle_boundary_condition_ = std::move(mantle_boundary_condition);

  // Validate boundary conditions
  using domain::BoundaryConditions::is_none;
  using domain::BoundaryConditions::is_periodic;
  if (lower_z_boundary_condition_ != nullptr) {
    if (is_none(lower_z_boundary_condition_)) {
      PARSE_ERROR(context,
                  "None boundary condition is not supported for LowerZ. "
                  "Use an outflow-type boundary condition instead.");
    }
    if (is_periodic(lower_z_boundary_condition_) xor
        is_periodic(upper_z_boundary_condition_)) {
      PARSE_ERROR(context,
                  "Either both lower and upper z-boundary conditions must "
                  "be periodic, or neither.");
    }
    if (is_periodic(lower_z_boundary_condition_) and
        is_periodic(upper_z_boundary_condition_)) {
      lower_z_boundary_condition_ = nullptr;
      upper_z_boundary_condition_ = nullptr;
    }
  }
  if (upper_z_boundary_condition_ != nullptr and
      is_none(upper_z_boundary_condition_)) {
    PARSE_ERROR(context,
                "None boundary condition is not supported for UpperZ. "
                "Use an outflow-type boundary condition instead.");
  }
  if (mantle_boundary_condition_ != nullptr) {
    if (is_none(mantle_boundary_condition_)) {
      PARSE_ERROR(context,
                  "None boundary condition is not supported for Mantle. "
                  "Use an outflow-type boundary condition instead.");
    }
    if (is_periodic(mantle_boundary_condition_)) {
      PARSE_ERROR(context,
                  "A cylinder can't have periodic boundary conditions in "
                  "the radial direction.");
    }
  } else {
    if (lower_z_boundary_condition_ != nullptr) {
      PARSE_ERROR(context,
                  "Mantle boundary condition is not set, but lower is. This "
                  "is probably a mistake");
    }
    if (upper_z_boundary_condition_ != nullptr) {
      PARSE_ERROR(context,
                  "Mantle boundary condition is not set, but upper z is. This "
                  "is probably a mistake");
    }
  }
}

Domain<3> UniformCylindricalFlatEndcap::create_domain() const {
  // TODO : change this to one coord map and not a vector since only one block
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coordinate_maps{};

  // Construct a coordinate map that goes from logical coordinates to a unit
  // right cylinder block. The radii and bounds are what are expected by the
  // UniformCylindricalFlatEndcap map.
  const double cylinder_inner_radius = 0.0;
  const double cylinder_outer_radius = 1.0;
  const double cylinder_lower_bound_z = -1.0;
  const double cylinder_upper_bound_z = 1.0;

  const auto logical_to_unit_cylinder_map =
      cyl_coordinate_map(cylinder_inner_radius, cylinder_outer_radius,
                         cylinder_lower_bound_z, cylinder_upper_bound_z);
  auto unit_cylinder_to_endcap_map =
      ::domain::CoordinateMaps::UniformCylindricalFlatEndcap(
          sphere_center_, cylinder_center_, sphere_radius_, cylinder_radius_,
          z_plane_);
  auto endcap_map = ::domain::push_back(logical_to_unit_cylinder_map,
                                        unit_cylinder_to_endcap_map);
  coordinate_maps.emplace_back(
      std::make_unique<std::decay_t<decltype(endcap_map)>>(
          std::move(endcap_map)));

  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);
  const size_t block_id = 0;
  blocks.emplace_back(std::move(coordinate_maps[block_id]), block_id,
                      DirectionMap<3, BlockNeighbors<3>>{},
                      block_names_.at(block_id),
                      ::domain::topologies::full_cylinder);

  Domain<3> domain{std::move(blocks), {}, block_groups_};

  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
UniformCylindricalFlatEndcap::external_boundary_conditions() const {
  if (mantle_boundary_condition_ == nullptr) {
    return {};
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{num_blocks_};

  const size_t block_id = 0;

  // Lower z boundary
  if (lower_z_boundary_condition_ != nullptr) {
    boundary_conditions[block_id][Direction<3>::lower_zeta()] =
        lower_z_boundary_condition_->get_clone();
  }

  // Upper z boundary
  if (upper_z_boundary_condition_ != nullptr) {
    boundary_conditions[block_id][Direction<3>::upper_zeta()] =
        upper_z_boundary_condition_->get_clone();
  }

  // Radial (mantle) boundary on the outermost radial block
  boundary_conditions[block_id][Direction<3>::upper_xi()] =
      mantle_boundary_condition_->get_clone();

  return boundary_conditions;
}

std::vector<std::array<size_t, 3>>
UniformCylindricalFlatEndcap::initial_extents() const {
  const std::vector<std::array<size_t, 3>> extents{
      {{{initial_radial_grid_points_, initial_theta_grid_points_,
         initial_z_grid_points_}}}};
  return extents;
}

std::vector<std::array<size_t, 3>>
UniformCylindricalFlatEndcap::initial_refinement_levels() const {
  // ZernikeB2 should never be refined.
  const std::vector<std::array<size_t, 3>> refinement_levels{
      {{{0, 0, initial_refinement_in_z_}}}};
  return refinement_levels;
}
}  // namespace domain::creators
