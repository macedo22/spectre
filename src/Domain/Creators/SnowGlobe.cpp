// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/SnowGlobe.hpp"

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
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
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
std::array<double, 3> rotate_from_z_to_x_axis(
    const std::array<double, 3> input) {
  return discrete_rotation(
      OrientationMap<3>{std::array<Direction<3>, 3>{Direction<3>::upper_zeta(),
                                                    Direction<3>::upper_eta(),
                                                    Direction<3>::lower_xi()}},
      input);
}
}  // namespace

namespace domain::creators {
SnowGlobe::SnowGlobe(const typename SphereCenter::type sphere_center,
                     const typename SphereInnerRadius::type sphere_inner_radius,
                     const typename SphereOuterRadius::type sphere_outer_radius,
                     const typename CylinderCenter::type cylinder_center,
                     const typename CylinderRadius::type cylinder_radius,
                     const typename ZPlane::type z_plane,
                     const typename RotationDirection::type direction,
                     const typename AreNeighbors::type are_neighbors,
                     const typename InitialCylinderRadialGridPoints::type
                         initial_cylinder_radial_grid_points,
                     const typename InitialCylinderThetaGridPoints::type
                         initial_cylinder_theta_grid_points,
                     const typename InitialCylinderZGridPoints::type
                         initial_cylinder_z_grid_points,
                     const typename InitialCylinderRefinementInZ::type
                         initial_cylinder_refinement_in_z,
                     const typename InitialSphereRadialGridPoints::type
                         initial_sphere_radial_grid_points,
                     const typename InitialSphereThetaGridPoints::type
                         initial_sphere_theta_grid_points,
                     const typename InitialSpherePhiGridPoints::type
                         initial_sphere_phi_grid_points,
                     const typename InitialSphereRefinementInR::type
                         initial_sphere_refinement_in_r,
                     const Options::Context& context)
    : sphere_center_(sphere_center),
      sphere_inner_radius_(sphere_inner_radius),
      sphere_outer_radius_(sphere_outer_radius),
      cylinder_center_(cylinder_center),
      cylinder_radius_(cylinder_radius),
      z_plane_(z_plane),
      direction_(direction),
      are_neighbors_(are_neighbors),
      initial_cylinder_radial_grid_points_(initial_cylinder_radial_grid_points),
      initial_cylinder_theta_grid_points_(initial_cylinder_theta_grid_points),
      initial_cylinder_z_grid_points_(initial_cylinder_z_grid_points),
      initial_cylinder_refinement_in_z_(initial_cylinder_refinement_in_z),
      initial_sphere_radial_grid_points_(initial_sphere_radial_grid_points),
      initial_sphere_theta_grid_points_(initial_sphere_theta_grid_points),
      initial_sphere_phi_grid_points_(initial_sphere_phi_grid_points),
      initial_sphere_refinement_in_r_(initial_sphere_refinement_in_r) {
  if (sphere_inner_radius_ <= 0.0) {
    PARSE_ERROR(context, "SphereInnerRadius must be positive, but is: "
                             << sphere_inner_radius_);
  }
  if (sphere_outer_radius_ <= sphere_inner_radius_) {
    PARSE_ERROR(context,
                "SphereOuterRadius must be greater than SphereInnerRadius, but "
                "SphereInnerRadius is "
                    << sphere_inner_radius_ << " and SphereOuterRadius is "
                    << sphere_outer_radius_);
  }

  if (cylinder_radius_ <= 0.0) {
    PARSE_ERROR(context, "CylinderRadius must be positive, but is: "
                             << cylinder_radius_);
  }

  if (z_plane_ <= sphere_center_[2]) {
    PARSE_ERROR(context,
                "ZPlane must be > z coordinate of SphereCenter, but ZPlane "
                "is "
                    << z_plane_ << " and SphereCenter z coordinate is "
                    << sphere_center_[2]);
  }
  if (z_plane_ >= sphere_center_[2] + sphere_outer_radius_) {
    PARSE_ERROR(
        context,
        "ZPlane must be < (SphereOuterRadius + z coordinate of SphereCenter), "
        "but ZPlane is "
            << z_plane_
            << " and (SphereOuterRadius + z coordinate of SphereCenter) is "
            << (sphere_center_[2] + sphere_outer_radius_));
  }
  if (z_plane_ >= cylinder_center_[2]) {
    PARSE_ERROR(
        context,
        "ZPlane must be < z coordinate of CylinderCenter, but ZPlane is "
            << z_plane_ << " and CylinderCenter z coordinate is "
            << cylinder_center_[2]);
  }

  if ((direction != 1) and (direction != 2) and (direction != 3) and
      (direction != -1) and (direction != -2) and (direction != -3)) {
    PARSE_ERROR(
        context,
        "RotationDirection must be one of {1, 2, 3, -1, -2, -3} but got "
            << direction);
  }

  if (initial_cylinder_theta_grid_points_ % 2 != 1) {
    PARSE_ERROR(context,
                "The number of cylinder theta grid points must be odd (this "
                "helps with numerical stability), but got "
                    << initial_cylinder_theta_grid_points_);
  }
  if (initial_cylinder_theta_grid_points_ >
      4 * initial_cylinder_radial_grid_points_ - 3) {
    PARSE_ERROR(context,
                "The number of cylinder theta grid points must be <= 4 * "
                "(the number of radial grid points) - 3, but got n_r = "
                    << initial_cylinder_radial_grid_points_
                    << "and n_theta = " << initial_cylinder_theta_grid_points_);
  }

  if (initial_sphere_theta_grid_points_ > initial_sphere_phi_grid_points_) {
    PARSE_ERROR(context,
                "The number of sphere theta grid points must be greater than "
                "the phi grid points, but got n_theta = "
                    << initial_sphere_theta_grid_points_
                    << " and n_phi = " << initial_sphere_phi_grid_points_);
  }
  if (initial_sphere_phi_grid_points_ % 2 != 1) {
    PARSE_ERROR(context,
                "The number of sphere phi grid points must be odd (this helps "
                "with numerical stability), but got "
                    << initial_sphere_phi_grid_points_);
  }
}

SnowGlobe::SnowGlobe(
    const typename SphereCenter::type sphere_center,
    const typename SphereInnerRadius::type sphere_inner_radius,
    const typename SphereOuterRadius::type sphere_outer_radius,
    const typename CylinderCenter::type cylinder_center,
    const typename CylinderRadius::type cylinder_radius,
    const typename ZPlane::type z_plane,
    const typename RotationDirection::type direction,
    const typename AreNeighbors::type are_neighbors,
    const typename InitialCylinderRadialGridPoints::type
        initial_cylinder_radial_grid_points,
    const typename InitialCylinderThetaGridPoints::type
        initial_cylinder_theta_grid_points,
    const typename InitialCylinderZGridPoints::type
        initial_cylinder_z_grid_points,
    const typename InitialCylinderRefinementInZ::type
        initial_cylinder_refinement_in_z,
    const typename InitialSphereRadialGridPoints::type
        initial_sphere_radial_grid_points,
    const typename InitialSphereThetaGridPoints::type
        initial_sphere_theta_grid_points,
    const typename InitialSpherePhiGridPoints::type
        initial_sphere_phi_grid_points,
    const typename InitialSphereRefinementInR::type
        initial_sphere_refinement_in_r,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        cylinder_lower_z_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        cylinder_upper_z_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        cylinder_mantle_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        sphere_inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        sphere_outer_boundary_condition,
    const Options::Context& context)
    : SnowGlobe(
          sphere_center, sphere_inner_radius, sphere_outer_radius,
          cylinder_center, cylinder_radius, z_plane, direction, are_neighbors,
          initial_cylinder_radial_grid_points,
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          initial_cylinder_refinement_in_z, initial_sphere_radial_grid_points,
          initial_sphere_theta_grid_points, initial_sphere_phi_grid_points,
          initial_sphere_refinement_in_r, context) {
  // NOLINTNEXTLINE
  cylinder_lower_z_boundary_condition_ =
      std::move(cylinder_lower_z_boundary_condition);
  // NOLINTNEXTLINE
  cylinder_upper_z_boundary_condition_ =
      std::move(cylinder_upper_z_boundary_condition);
  // NOLINTNEXTLINE
  cylinder_mantle_boundary_condition_ =
      std::move(cylinder_mantle_boundary_condition);
  // NOLINTNEXTLINE
  sphere_inner_boundary_condition_ = std::move(sphere_inner_boundary_condition);
  // NOLINTNEXTLINE
  sphere_outer_boundary_condition_ = std::move(sphere_outer_boundary_condition);

  // Validate boundary conditions
  using domain::BoundaryConditions::is_none;
  using domain::BoundaryConditions::is_periodic;
  if (cylinder_lower_z_boundary_condition_ != nullptr) {
    if (not are_neighbors_ and is_none(cylinder_lower_z_boundary_condition_)) {
      PARSE_ERROR(context,
                  "None boundary condition is not supported for CylinderLowerZ "
                  "when AreNeighbors = false. "
                  "Use an outflow-type boundary condition instead.");
    }
    if (is_periodic(cylinder_lower_z_boundary_condition_) xor
        is_periodic(cylinder_upper_z_boundary_condition_)) {
      PARSE_ERROR(context,
                  "Either both lower and upper z-boundary conditions must "
                  "be periodic, or neither.");
    }
    if (are_neighbors_ and not is_none(cylinder_lower_z_boundary_condition_)) {
      PARSE_ERROR(context,
                  "Boundary condition for CylinderLowerZ should be None when "
                  "AreNeighbors = true.");
    }
  }
  if (cylinder_upper_z_boundary_condition_ != nullptr) {
    if (is_none(cylinder_upper_z_boundary_condition_)) {
      PARSE_ERROR(
          context,
          "None boundary condition is not supported for CylinderUpperZ. "
          "Use an outflow-type boundary condition instead.");
    }
  }
  if (cylinder_mantle_boundary_condition_ != nullptr) {
    if (is_none(cylinder_mantle_boundary_condition_)) {
      PARSE_ERROR(context,
                  "None boundary condition is not supported for Mantle. "
                  "Use an outflow-type boundary condition instead.");
    }
    if (is_periodic(cylinder_mantle_boundary_condition_)) {
      PARSE_ERROR(context,
                  "A cylinder can't have periodic boundary conditions in "
                  "the radial direction.");
    }
  } else {
    if (cylinder_lower_z_boundary_condition_ != nullptr) {
      PARSE_ERROR(context,
                  "Mantle boundary condition is not set, but lower is. This "
                  "is probably a mistake");
    }
    if (cylinder_upper_z_boundary_condition_ != nullptr) {
      PARSE_ERROR(context,
                  "Mantle boundary condition is not set, but upper is. This "
                  "is probably a mistake");
    }
  }
}

Domain<3> SnowGlobe::create_domain() const {
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coordinate_maps{};

  const OrientationMap<3> aligned = OrientationMap<3>::create_aligned();

  const OrientationMap<3> rotate_to_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
      Direction<3>::lower_xi()}};
  const OrientationMap<3> rotate_to_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
      Direction<3>::lower_eta()}};
  const OrientationMap<3> rotate_to_z_axis = aligned;
  const OrientationMap<3> rotate_to_minus_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
      Direction<3>::upper_xi()}};
  const OrientationMap<3> rotate_to_minus_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
      Direction<3>::upper_eta()}};
  const OrientationMap<3> rotate_to_minus_z_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::lower_eta(),
      Direction<3>::lower_zeta()}};

  OrientationMap<3> cylinder_orientation_map;
  if (direction_ == -1) {
    cylinder_orientation_map = rotate_to_minus_x_axis;
  } else if (direction_ == -2) {
    cylinder_orientation_map = rotate_to_minus_y_axis;
  } else if (direction_ == -3) {
    cylinder_orientation_map = rotate_to_minus_z_axis;
  } else if (direction_ == 1) {
    cylinder_orientation_map = rotate_to_x_axis;
  } else if (direction_ == 2) {
    cylinder_orientation_map = rotate_to_y_axis;
  } else {
    // direction_ == 3
    cylinder_orientation_map = rotate_to_z_axis;
  }

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

  // Add coordinate map for cylinder
  auto unit_cylinder_to_endcap_map =
      ::domain::CoordinateMaps::UniformCylindricalFlatEndcap(
          sphere_center_, cylinder_center_, sphere_outer_radius_,
          cylinder_radius_, z_plane_);
  auto endcap_map = ::domain::push_back(
      ::domain::push_back(logical_to_unit_cylinder_map,
                          unit_cylinder_to_endcap_map),
      CoordinateMaps::DiscreteRotation<3>(cylinder_orientation_map));
  coordinate_maps.emplace_back(
      std::make_unique<std::decay_t<decltype(endcap_map)>>(
          std::move(endcap_map)));

  using Affine = ::domain::CoordinateMaps::Affine;
  auto make_spherical_shell_coord_map =
      [](const double inner_radius, const double outer_radius,
         const std::array<double, 3>& aligned_center) {
        CoordinateMaps::Interval radial_map{
            -1.0,
            1.0,
            inner_radius,
            outer_radius,
            ::domain::CoordinateMaps::Distribution::Linear,
            0.0};
        return make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::Interval,
                                           CoordinateMaps::Identity<2>>{
                std::move(radial_map), CoordinateMaps::Identity<2>{}},
            CoordinateMaps::SphericalToCartesianPfaffian{},
            CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>{
                Affine{-1.0, 1.0, -1.0 + aligned_center[0],
                       1.0 + aligned_center[0]},
                Affine{-1.0, 1.0, -1.0 + aligned_center[1],
                       1.0 + aligned_center[1]},
                Affine{-1.0, 1.0, -1.0 + aligned_center[2],
                       1.0 + aligned_center[2]}});
      };

  auto sphere_map =
      make_spherical_shell_coord_map(sphere_inner_radius_, sphere_outer_radius_,
                                     rotate_from_z_to_x_axis(sphere_center_));

  // Make the cylinder and sphere neighbors of each other
  std::vector<DirectionMap<3, BlockNeighbors<3>>> neighbors(num_blocks_);

  const size_t cylinder_block = 0;
  const size_t sphere_block = 1;

  if (are_neighbors_) {
    neighbors[cylinder_block].emplace(
        Direction<3>::lower_zeta(),
        BlockNeighbors<3>{
            {sphere_block},
            {{sphere_block,
              OrientationMap<3>{{{Direction<3>::self(), Direction<3>::self(),
                                  Direction<3>::upper_xi()}}}}},
            /*are_conforming=*/false});
    neighbors[sphere_block].emplace(
        Direction<3>::upper_xi(),
        BlockNeighbors<3>{
            {cylinder_block},
            {{cylinder_block, OrientationMap<3>{{{Direction<3>::upper_zeta(),
                                                  Direction<3>::self(),
                                                  Direction<3>::self()}}}}},
            /*are_conforming=*/false});

    const bool first_inverse_check =
        neighbors[sphere_block][Direction<3>::upper_xi()].orientations().at(
            cylinder_block) ==
            neighbors[cylinder_block][Direction<3>::lower_zeta()]
                .orientations()
                .at(sphere_block)
                .inverse_map();
    
    const bool second_inverse_check =
        neighbors[cylinder_block][Direction<3>::lower_zeta()].orientations().at(
            sphere_block) == neighbors[sphere_block][Direction<3>::upper_xi()]
                                 .orientations()
                                 .at(cylinder_block)
                                 .inverse_map();
    
    const std::string error_msg{"The cylinder to shell and shell to cylinder neighbor maps are not inverses of each other."};

    if (not first_inverse_check) {
      ERROR(error_msg);
    }
    if (not second_inverse_check) {
      ERROR(error_msg);
    }
  }

  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);

  blocks.emplace_back(std::move(coordinate_maps[cylinder_block]),
                      cylinder_block, std::move(neighbors[cylinder_block]),
                      block_names_.at(cylinder_block),
                      ::domain::topologies::full_cylinder);
  blocks.emplace_back(
      std::move(sphere_map), sphere_block, std::move(neighbors[sphere_block]),
      block_names_.at(sphere_block), ::domain::topologies::spherical_shell);

  Domain<3> domain{std::move(blocks), {}, block_groups_};

  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
SnowGlobe::external_boundary_conditions() const {
  if (cylinder_mantle_boundary_condition_ == nullptr) {
    return {};
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions(num_blocks_);

  const size_t cylinder_block = 0;
  const size_t sphere_block = 1;

  // Lower z boundary
  if (not are_neighbors_ and cylinder_lower_z_boundary_condition_ != nullptr) {
    boundary_conditions[cylinder_block][Direction<3>::lower_zeta()] =
        cylinder_lower_z_boundary_condition_->get_clone();
  }

  // Upper z boundary
  if (cylinder_upper_z_boundary_condition_ != nullptr) {
    boundary_conditions[cylinder_block][Direction<3>::upper_zeta()] =
        cylinder_upper_z_boundary_condition_->get_clone();
  }

  // Radial (mantle) boundary on the outermost radial block
  boundary_conditions[cylinder_block][Direction<3>::upper_xi()] =
      cylinder_mantle_boundary_condition_->get_clone();

  // Sphere inner boundary
  boundary_conditions[sphere_block][Direction<3>::lower_xi()] =
      sphere_inner_boundary_condition_->get_clone();

  // Sphere outer boundary
  boundary_conditions[sphere_block][Direction<3>::upper_xi()] =
      sphere_outer_boundary_condition_->get_clone();

  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> SnowGlobe::initial_extents() const {
  const std::vector<std::array<size_t, 3>> extents{
      {{{initial_cylinder_radial_grid_points_,
         initial_cylinder_theta_grid_points_, initial_cylinder_z_grid_points_}},
       {{initial_sphere_radial_grid_points_, initial_sphere_theta_grid_points_,
         initial_sphere_phi_grid_points_}}}};
  return extents;
}

std::vector<std::array<size_t, 3>> SnowGlobe::initial_refinement_levels()
    const {
  // Spherical harmonics and ZernikeB2 should never be refined.
  const std::vector<std::array<size_t, 3>> refinement_levels{
      {{{0, 0, initial_cylinder_refinement_in_z_}},
       {{initial_sphere_refinement_in_r_, 0, 0}}}};
  return refinement_levels;
}
}  // namespace domain::creators
