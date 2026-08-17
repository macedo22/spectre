// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/AbuttingUniformCylindricalFlatEndcaps.hpp"

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
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
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace {
std::array<double, 3> flip_about_xy_plane(const std::array<double, 3> input) {
  return std::array<double, 3>{input[0], input[1], -input[2]};
}
}  // namespace

namespace domain::creators {
AbuttingUniformCylindricalFlatEndcaps::AbuttingUniformCylindricalFlatEndcaps(
    const typename SphereCenterA::type sphere_center_a,
    const typename SphereCenterB::type sphere_center_b,
    const typename SphereRadiusA::type sphere_radius_a,
    const typename SphereRadiusB::type sphere_radius_b,
    const typename CylinderCenter::type cylinder_center,
    const typename CylinderRadius::type cylinder_radius,
    const typename ZPlaneA::type z_plane_a,
    const typename ZPlaneB::type z_plane_b,
    const typename RotationDirection::type direction,
    const typename AlignNodesOnSharedFace::type aligned_nodes_on_shared_face,
    const typename AreNeighbors::type are_neighbors,
    const typename InitialRadialGridPointsA::type initial_radial_grid_points_a,
    const typename InitialRadialGridPointsB::type initial_radial_grid_points_b,
    const typename InitialThetaGridPointsA::type initial_theta_grid_points_a,
    const typename InitialThetaGridPointsB::type initial_theta_grid_points_b,
    const typename InitialZGridPointsA::type initial_z_grid_points_a,
    const typename InitialZGridPointsB::type initial_z_grid_points_b,
    const typename InitialRefinementInZA::type initial_refinement_in_z_a,
    const typename InitialRefinementInZB::type initial_refinement_in_z_b,
    const Options::Context& context)
    : sphere_center_a_(sphere_center_a),
      sphere_center_b_(sphere_center_b),
      sphere_radius_a_(sphere_radius_a),
      sphere_radius_b_(sphere_radius_b),
      cylinder_center_(cylinder_center),
      cylinder_radius_(cylinder_radius),
      z_plane_a_(z_plane_a),
      z_plane_b_(z_plane_b),
      direction_(direction),
      aligned_nodes_on_shared_face_(aligned_nodes_on_shared_face),
      are_neighbors_(are_neighbors),
      initial_radial_grid_points_a_(initial_radial_grid_points_a),
      initial_radial_grid_points_b_(initial_radial_grid_points_b),
      initial_theta_grid_points_a_(initial_theta_grid_points_a),
      initial_theta_grid_points_b_(initial_theta_grid_points_b),
      initial_z_grid_points_a_(initial_z_grid_points_a),
      initial_z_grid_points_b_(initial_z_grid_points_b),
      initial_refinement_in_z_a_(initial_refinement_in_z_a),
      initial_refinement_in_z_b_(initial_refinement_in_z_b) {
  if (sphere_radius_a_ <= 0.0) {
    PARSE_ERROR(context,
                "SphereRadiusA must be positive, but is: " << sphere_radius_a_);
  }
  if (sphere_radius_b_ <= 0.0) {
    PARSE_ERROR(context,
                "SphereRadiusB must be positive, but is: " << sphere_radius_b_);
  }

  if (cylinder_radius_ <= 0.0) {
    PARSE_ERROR(context, "CylinderRadius must be positive, but is: "
                             << cylinder_radius_);
  }

  if (z_plane_a_ >= sphere_center_a_[2]) {
    PARSE_ERROR(context,
                "ZPlaneA must be < z coordinate of SphereCenterA, but ZPlaneA "
                "is "
                    << z_plane_a_ << " and SphereCenterA z coordinate is "
                    << sphere_center_a_[2]);
  }
  if (z_plane_b_ <= sphere_center_b_[2]) {
    PARSE_ERROR(context,
                "ZPlaneB must be > z coordinate of SphereCenterB, but ZPlaneB "
                "is "
                    << z_plane_b_ << " and SphereCenterB z coordinate is "
                    << sphere_center_b_[2]);
  }
  if (z_plane_a_ <= sphere_center_a_[2] - sphere_radius_a_) {
    PARSE_ERROR(context,
                "ZPlaneA must be > (SphereRadiusA - z coordinate of "
                "SphereCenterA), but ZPlaneA is "
                    << z_plane_a_
                    << " and (SphereRadiusA - z coordinate of SphereCenterA) "
                       "is "
                    << (sphere_center_a_[2] - sphere_radius_a_));
  }
  if (z_plane_b_ >= sphere_center_b_[2] + sphere_radius_b_) {
    PARSE_ERROR(context,
                "ZPlaneB must be < (SphereRadiusB + z coordinate of "
                "SphereCenterB), but ZPlaneB is "
                    << z_plane_b_
                    << " and (SphereRadiusB + z coordinate of SphereCenterB) "
                       "is "
                    << (sphere_center_b_[2] + sphere_radius_b_));
  }
  if (z_plane_a_ <= cylinder_center_[2]) {
    PARSE_ERROR(
        context,
        "ZPlaneA must be > z coordinate of CylinderCenter, but ZPlaneA is "
            << z_plane_a_ << " and CylinderCenter z coordinate is "
            << cylinder_center_[2]);
  }
  if (z_plane_b_ >= cylinder_center_[2]) {
    PARSE_ERROR(
        context,
        "ZPlaneB must be < z coordinate of CylinderCenter, but ZPlaneB is "
            << z_plane_b_ << " and CylinderCenter z coordinate is "
            << cylinder_center_[2]);
  }

  if (direction != 1) {
    PARSE_ERROR(context,
                "RotationDirection must be 1 because only +x supported for "
                "now, but got "
                    << direction);
  }

  if (initial_theta_grid_points_a_ % 2 != 1) {
    PARSE_ERROR(context,
                "The number of angular grid points must be odd (this helps "
                "with numerical stability), but got "
                    << initial_theta_grid_points_a_);
  }
  if (initial_theta_grid_points_b_ % 2 != 1) {
    PARSE_ERROR(context,
                "The number of angular grid points must be odd (this helps "
                "with numerical stability), but got "
                    << initial_theta_grid_points_b_);
  }
  if (initial_theta_grid_points_a_ > 4 * initial_radial_grid_points_a_ - 3) {
    PARSE_ERROR(context,
                "The number of angular grid points must be <= 4 * "
                "(the number of radial grid points) - 3, but got n_r = "
                    << initial_radial_grid_points_a_
                    << "and n_theta = " << initial_theta_grid_points_a_);
  }
  if (initial_theta_grid_points_b_ > 4 * initial_radial_grid_points_b_ - 3) {
    PARSE_ERROR(context,
                "The number of angular grid points must be <= 4 * "
                "(the number of radial grid points) - 3, but got n_r = "
                    << initial_radial_grid_points_b_
                    << "and n_theta = " << initial_theta_grid_points_b_);
  }
}

AbuttingUniformCylindricalFlatEndcaps::AbuttingUniformCylindricalFlatEndcaps(
    const typename SphereCenterA::type sphere_center_a,
    const typename SphereCenterB::type sphere_center_b,
    const typename SphereRadiusA::type sphere_radius_a,
    const typename SphereRadiusB::type sphere_radius_b,
    const typename CylinderCenter::type cylinder_center,
    const typename CylinderRadius::type cylinder_radius,
    const typename ZPlaneA::type z_plane_a,
    const typename ZPlaneB::type z_plane_b,
    const typename RotationDirection::type direction,
    const typename AlignNodesOnSharedFace::type aligned_nodes_on_shared_face,
    const typename AreNeighbors::type are_neighbors,
    const typename InitialRadialGridPointsA::type initial_radial_grid_points_a,
    const typename InitialRadialGridPointsB::type initial_radial_grid_points_b,
    const typename InitialThetaGridPointsA::type initial_theta_grid_points_a,
    const typename InitialThetaGridPointsB::type initial_theta_grid_points_b,
    const typename InitialZGridPointsA::type initial_z_grid_points_a,
    const typename InitialZGridPointsB::type initial_z_grid_points_b,
    const typename InitialRefinementInZA::type initial_refinement_in_z_a,
    const typename InitialRefinementInZB::type initial_refinement_in_z_b,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        lower_z_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        mantle_boundary_condition,
    const Options::Context& context)
    : AbuttingUniformCylindricalFlatEndcaps(
          sphere_center_a, sphere_center_b, sphere_radius_a, sphere_radius_b,
          cylinder_center, cylinder_radius, z_plane_a, z_plane_b, direction,
          aligned_nodes_on_shared_face, are_neighbors,
          initial_radial_grid_points_a, initial_radial_grid_points_b,
          initial_theta_grid_points_a, initial_theta_grid_points_b,
          initial_z_grid_points_a, initial_z_grid_points_b,
          initial_refinement_in_z_a, initial_refinement_in_z_b, context) {
  // NOLINTNEXTLINE
  lower_z_boundary_condition_ = std::move(lower_z_boundary_condition);
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
  }
}

Domain<3> AbuttingUniformCylindricalFlatEndcaps::create_domain() const {
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coordinate_maps{};

  const OrientationMap<3> rotate_to_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
      Direction<3>::lower_xi()}};
  const OrientationMap<3> rotate_to_minus_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
      Direction<3>::upper_xi()}};

  const OrientationMap<3> aligned = OrientationMap<3>::create_aligned();

  // 180 degree rotation about a cylinder's z axis
  const OrientationMap<3> half_turn_about_zeta{std::array<Direction<3>, 3>{
      Direction<3>::lower_xi(), Direction<3>::lower_eta(),
      Direction<3>::upper_zeta()}};

  const OrientationMap<3> pre_rotation_map_a =
      aligned_nodes_on_shared_face_ ? half_turn_about_zeta : aligned;
  const OrientationMap<3> pre_rotation_map_b = aligned;

  ASSERT(direction_ == 1,
         "Only +x direction supported for now. Use the value 1 for direction.");

  const OrientationMap<3> orientation_map_a = rotate_to_minus_x_axis;
  const OrientationMap<3> orientation_map_b = rotate_to_x_axis;

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

  // Add coordinate map for first cylinder
  auto unit_cylinder_to_endcap_map_a =
      ::domain::CoordinateMaps::UniformCylindricalFlatEndcap(
          flip_about_xy_plane(sphere_center_a_),
          flip_about_xy_plane(cylinder_center_), sphere_radius_a_,
          cylinder_radius_, -z_plane_a_);
  auto endcap_map_a = ::domain::push_back(
      ::domain::push_back(
          ::domain::push_back(
              logical_to_unit_cylinder_map,
              CoordinateMaps::DiscreteRotation<3>(pre_rotation_map_a)),
          unit_cylinder_to_endcap_map_a),
      CoordinateMaps::DiscreteRotation<3>(orientation_map_a));
  coordinate_maps.emplace_back(
      std::make_unique<std::decay_t<decltype(endcap_map_a)>>(
          std::move(endcap_map_a)));

  // Add coordinate map for second cylinder
  auto unit_cylinder_to_endcap_map_b =
      ::domain::CoordinateMaps::UniformCylindricalFlatEndcap(
          sphere_center_b_, cylinder_center_, sphere_radius_b_,
          cylinder_radius_, z_plane_b_);
  auto endcap_map_b = ::domain::push_back(
      ::domain::push_back(
          ::domain::push_back(
              logical_to_unit_cylinder_map,
              CoordinateMaps::DiscreteRotation<3>(pre_rotation_map_b)),
          unit_cylinder_to_endcap_map_b),
      CoordinateMaps::DiscreteRotation<3>(orientation_map_b));
  coordinate_maps.emplace_back(
      std::make_unique<std::decay_t<decltype(endcap_map_b)>>(
          std::move(endcap_map_b)));

  // Make the two cylinders neighbors of each other
  std::vector<DirectionMap<3, BlockNeighbors<3>>> neighbors(
      coordinate_maps.size());

  const size_t block_id_a = 0;
  const size_t block_id_b = 1;

  if (are_neighbors_) {
    neighbors[block_id_a].emplace(
        Direction<3>::upper_zeta(),
        BlockNeighbors<3>{
            {block_id_b},
            {{block_id_b, OrientationMap<3>{{{Direction<3>::upper_xi(),
                                              Direction<3>::lower_eta(),
                                              Direction<3>::lower_zeta()}}}}},
            /*are_conforming=*/aligned_nodes_on_shared_face_});
    neighbors[block_id_b].emplace(
        Direction<3>::upper_zeta(),
        BlockNeighbors<3>{
            {block_id_a},
            {{block_id_a, OrientationMap<3>{{{Direction<3>::upper_xi(),
                                              Direction<3>::lower_eta(),
                                              Direction<3>::lower_zeta()}}}}},
            /*are_conforming=*/aligned_nodes_on_shared_face_});
  }

  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);

  blocks.emplace_back(std::move(coordinate_maps[block_id_a]), block_id_a,
                      std::move(neighbors[block_id_a]),
                      block_names_.at(block_id_a),
                      ::domain::topologies::full_cylinder);
  blocks.emplace_back(std::move(coordinate_maps[block_id_b]), block_id_b,
                      std::move(neighbors[block_id_b]),
                      block_names_.at(block_id_b),
                      ::domain::topologies::full_cylinder);

  Domain<3> domain{std::move(blocks), {}, block_groups_};

  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
AbuttingUniformCylindricalFlatEndcaps::external_boundary_conditions() const {
  if (mantle_boundary_condition_ == nullptr) {
    return {};
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions(num_blocks_);

  const size_t block_id_a = 0;
  const size_t block_id_b = 1;

  // Lower z boundary
  if (lower_z_boundary_condition_ != nullptr) {
    boundary_conditions[block_id_a][Direction<3>::lower_zeta()] =
        lower_z_boundary_condition_->get_clone();
    boundary_conditions[block_id_b][Direction<3>::lower_zeta()] =
        lower_z_boundary_condition_->get_clone();
  }

  // Upper z boundary only if blocks aren't neighbors
  if (not are_neighbors_ and lower_z_boundary_condition_ != nullptr) {
    boundary_conditions[block_id_a][Direction<3>::upper_zeta()] =
        lower_z_boundary_condition_->get_clone();
    boundary_conditions[block_id_b][Direction<3>::upper_zeta()] =
        lower_z_boundary_condition_->get_clone();
  }

  // Radial (mantle) boundary on the outermost radial block
  boundary_conditions[block_id_a][Direction<3>::upper_xi()] =
      mantle_boundary_condition_->get_clone();
  boundary_conditions[block_id_b][Direction<3>::upper_xi()] =
      mantle_boundary_condition_->get_clone();

  return boundary_conditions;
}

std::vector<std::array<size_t, 3>>
AbuttingUniformCylindricalFlatEndcaps::initial_extents() const {
  const std::vector<std::array<size_t, 3>> extents{
      {{{initial_radial_grid_points_a_, initial_theta_grid_points_a_,
         initial_z_grid_points_a_}},
       {{initial_radial_grid_points_b_, initial_theta_grid_points_b_,
         initial_z_grid_points_b_}}}};
  return extents;
}

std::vector<std::array<size_t, 3>>
AbuttingUniformCylindricalFlatEndcaps::initial_refinement_levels() const {
  // ZernikeB2 should never be refined.
  const std::vector<std::array<size_t, 3>> refinement_levels{
      {{{0, 0, initial_refinement_in_z_a_}},
       {{0, 0, initial_refinement_in_z_b_}}}};
  return refinement_levels;
}
}  // namespace domain::creators
