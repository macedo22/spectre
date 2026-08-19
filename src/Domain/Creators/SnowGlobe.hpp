// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
template <size_t VolumeDim>
class DiscreteRotation;
template <size_t Dim>
class Identity;
class Interval;
class PolarToCartesian;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
class SphericalToCartesianPfaffian;
class UniformCylindricalFlatEndcap;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {
/*!
 * \brief Create a domain resembling a snow globe, consisting of a spherical
 * shell block and a flat cylindrical endcap. This is meant to replicate the
 * inner spherical shell and M filled cylinder in
 * `::domain::creators::CylindricalBinaryCompactObject`.
 */
class SnowGlobe : public DomainCreator<3> {
 public:
  using unit_cylinder_map =
      CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                     CoordinateMaps::Identity<1>,
                                     CoordinateMaps::Interval>;
  using polar_to_cartesian_map =
      CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                     CoordinateMaps::Identity<1>>;

  using maps_list = tmpl::flatten<tmpl::list<
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            unit_cylinder_map, polar_to_cartesian_map,
                            CoordinateMaps::UniformCylindricalFlatEndcap,
                            CoordinateMaps::DiscreteRotation<3>>,
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          domain::CoordinateMaps::ProductOf2Maps<CoordinateMaps::Interval,
                                                 CoordinateMaps::Identity<2>>,
          domain::CoordinateMaps::SphericalToCartesianPfaffian,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Affine,
                                         CoordinateMaps::Affine>>>>;

  /*!
   * \brief Center of the sphere that abuts the lower surface of the
   * cylindrical endcap
   */
  struct SphereCenter {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center of the sphere that abuts the lower surface of the cylindrical "
        "endcap. See domain::CoordinateMaps::UniformCylindricalFlatEndcap for "
        "more details."};
  };

  /*!
   * \brief Radius of the inner edge of the spherical shell that abuts the lower
   * surface of the cylindrical endcap.
   */
  struct SphereInnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Inner radius of the spherical shell that abuts the lower surface of "
        "the cylindrical endcap. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Radius of the outer edge of the spherical shell that abuts the upper
   * surface of the cylindrical endcap.
   */
  struct SphereOuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Outer radius of the spherical shell that abuts the lower surface of "
        "the cylindrical endcap. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Center of the cylinder's upper face
   */
  struct CylinderCenter {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center of the cylinder's upper face. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Radius of the cylinder's upper face
   */
  struct CylinderRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the cylinder's upper face."};
  };

  /*!
   * \brief Plane of intersection between the sphere and cylindrical endcap.
   * See `domain::CoordinateMaps::UniformCylindricalFlatEndcap` for more
   * details.
   */
  struct ZPlane {
    using type = double;
    static constexpr Options::String help = {
        "Plane of intersection between the sphere and cylindrical endcap. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Direction of the normal to the flat cylindrical face of the endcap
   */
  struct RotationDirection {
    using type = int;
    static constexpr Options::String help = {
        "Direction of the normal to the flat cylindrical face of the endcap: "
        "+x: +1, +y: +2, +z: +3, -x: -1, -y: -2, -z: -3. Specify +x for the "
        "orientation of InnerSphereB and MBFilledCylinder in the "
        "CylindricalBinaryCompactObject. Specify +z for an unrotated, "
        "upside-down snowglobe. Specify -z for a proper upright snowglobe for "
        "your desk."};
  };

  /*!
   * \brief Whether or not the sphere and cylinder are designated as block
   * neighbors or two standalone blocks.
   */
  struct AreNeighbors {
    using type = bool;
    static constexpr Options::String help = {
        "Whether or not the cylinder and sphere are designated as block "
        "neighbors or two standalone blocks."};
  };

  /*!
   * \brief Initial number of radial gridpoints for the cylinder.
   */
  struct InitialCylinderRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for the cylinder."};
  };

  /*!
   * \brief Initial number of \f$\theta\f$ gridpoints for the cylinder. This is
   * enforced to be odd for numerical stability.
   */
  struct InitialCylinderThetaGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in theta for the cylinder. It must be "
        "an odd number for stability. "
        "It must be <= 4 * (initial radial grid points) - 3."};
  };

  /*!
   * \brief Initial number of z gridpoints for the cylinder
   */
  struct InitialCylinderZGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in z for the cylinder."};
  };

  /*!
   * \brief Initial refinement level in the z direction for the cylinder
   */
  struct InitialCylinderRefinementInZ {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in the z-direction for the cylinder."};
  };

  /*!
   * \brief Initial number of radial gridpoints for the sphere.
   */
  struct InitialSphereRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for the sphere."};
  };

  /*!
   * \brief Initial number of \f$\theta\f$ gridpoints for the sphere. This is
   * enforced to be odd for numerical stability.
   */
  struct InitialSphereThetaGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in theta for the sphere. It must be "
        "an odd number for stability. "
        "It must be <= 4 * (initial radial grid points) - 3."};
  };

  /*!
   * \brief Initial number of \f$\phi\f$ gridpoints for the sphere. This is
   * enforced to be odd for numerical stability.
   */
  struct InitialSpherePhiGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in phi for the sphere."};
  };

  /*!
   * \brief Initial refinement level in the radial direction for the sphere
   */
  struct InitialSphereRefinementInR {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in the radial direction for the sphere."};
  };

  /*!
   * \brief Boundary conditions group
   */
  struct BoundaryConditions {
    static constexpr Options::String help =
        "Options for the boundary conditions";
  };

  /*!
   * \brief Boundary condition on the lower base of the cylinders
   */
  template <typename BoundaryConditionsBase>
  struct CylinderLowerZBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "CylinderLowerZ"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the lower spherical base of "
        "the cylinder. Note that if you imagine an upright snowglobe, this is "
        "actually the upper face of the cylindrical base in that picture. In "
        "other words, it is the face that abuts the sphere. This should be set "
        "to None when AreNeighbors == true.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Boundary condition on the upper base of the cylinders
   */
  template <typename BoundaryConditionsBase>
  struct CylinderUpperZBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "CylinderUpperZ"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the upper spherical base of "
        "the cylinder. Note that if you imagine an upright snowglobe, this is "
        "actually the lower face of the cylindrical base in that picture. In "
        "other words, it is the face that does not abut the sphere.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Boundary condition on the radial boundary of the cylinder
   */
  template <typename BoundaryConditionsBase>
  struct CylinderMantleBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "CylinderMantle"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the mantle of the cylinder, "
        "i.e. at the `CylinderRadius` in the radial direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Boundary condition on the inner boundary of the sphere
   */
  template <typename BoundaryConditionsBase>
  struct SphereInnerBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "SphereInner"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the inner boundary of the "
        "sphere.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Boundary condition on the outer boundary of the sphere
   */
  template <typename BoundaryConditionsBase>
  struct SphereOuterBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "SphereOuter"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the outer boundary of the "
        "sphere.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options =
      tmpl::list<SphereCenter, SphereInnerRadius, SphereOuterRadius,
                 CylinderCenter, CylinderRadius, ZPlane, RotationDirection,
                 AreNeighbors, InitialCylinderRadialGridPoints,
                 InitialCylinderThetaGridPoints, InitialCylinderZGridPoints,
                 InitialCylinderRefinementInZ, InitialSphereRadialGridPoints,
                 InitialSphereThetaGridPoints, InitialSpherePhiGridPoints,
                 InitialSphereRefinementInR>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::append<
          basic_options,
          tmpl::list<
              CylinderLowerZBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              CylinderUpperZBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              CylinderMantleBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              SphereInnerBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              SphereOuterBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>>>,
      basic_options>;

  static constexpr Options::String help{
      "Creates two a snow globe shape from a spherical shell block and a flat "
      "cylindrical endcap block. See "
      "`::domain::CoordinateMaps::UniformCylindricalFlatEndcap` for details "
      "and to help with setting the cylinder input parameters."};

  SnowGlobe(
      typename SphereCenter::type sphere_center,
      typename SphereInnerRadius::type sphere_inner_radius,
      typename SphereOuterRadius::type sphere_outer_radius,
      typename CylinderCenter::type cylinder_center,
      typename CylinderRadius::type cylinder_radius,
      typename ZPlane::type z_plane, typename RotationDirection::type direction,
      typename AreNeighbors::type are_neighbors,
      typename InitialCylinderRadialGridPoints::type
          initial_cylinder_radial_grid_points,
      typename InitialCylinderThetaGridPoints::type
          initial_cylinder_theta_grid_points,
      typename InitialCylinderZGridPoints::type initial_cylinder_z_grid_points,
      typename InitialCylinderRefinementInZ::type
          initial_cylinder_refinement_in_z,
      typename InitialSphereRadialGridPoints::type
          initial_sphere_radial_grid_points,
      typename InitialSphereThetaGridPoints::type
          initial_sphere_theta_grid_points,
      typename InitialSpherePhiGridPoints::type initial_sphere_phi_grid_points,
      typename InitialSphereRefinementInR::type initial_sphere_refinement_in_r,
      const Options::Context& context = {});

  SnowGlobe(
      typename SphereCenter::type sphere_center,
      typename SphereInnerRadius::type sphere_inner_radius,
      typename SphereOuterRadius::type sphere_outer_radius,
      typename CylinderCenter::type cylinder_center,
      typename CylinderRadius::type cylinder_radius,
      typename ZPlane::type z_plane, typename RotationDirection::type direction,
      typename AreNeighbors::type are_neighbors,
      typename InitialCylinderRadialGridPoints::type
          initial_cylinder_radial_grid_points,
      typename InitialCylinderThetaGridPoints::type
          initial_cylinder_theta_grid_points,
      typename InitialCylinderZGridPoints::type initial_cylinder_z_grid_points,
      typename InitialCylinderRefinementInZ::type
          initial_cylinder_refinement_in_z,
      typename InitialSphereRadialGridPoints::type
          initial_sphere_radial_grid_points,
      typename InitialSphereThetaGridPoints::type
          initial_sphere_theta_grid_points,
      typename InitialSpherePhiGridPoints::type initial_sphere_phi_grid_points,
      typename InitialSphereRefinementInR::type initial_sphere_refinement_in_r,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          cylinder_lower_z_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          cylinder_upper_z_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          cylinder_mantle_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          sphere_inner_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          sphere_outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  SnowGlobe() = default;
  SnowGlobe(const SnowGlobe&) = delete;
  SnowGlobe(SnowGlobe&&) = default;
  SnowGlobe& operator=(const SnowGlobe&) = delete;
  SnowGlobe& operator=(SnowGlobe&&) = default;
  ~SnowGlobe() override = default;

  Domain<3> create_domain() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

 private:
  typename SphereCenter::type sphere_center_{};
  typename SphereInnerRadius::type sphere_inner_radius_{};
  typename SphereOuterRadius::type sphere_outer_radius_{};
  typename CylinderCenter::type cylinder_center_{};
  typename CylinderRadius::type cylinder_radius_{};
  typename ZPlane::type z_plane_{};
  typename RotationDirection::type direction_{};
  typename AreNeighbors::type are_neighbors_{};
  typename InitialCylinderRadialGridPoints::type
      initial_cylinder_radial_grid_points_{};
  typename InitialCylinderThetaGridPoints::type
      initial_cylinder_theta_grid_points_{};
  typename InitialCylinderZGridPoints::type initial_cylinder_z_grid_points_{};
  typename InitialCylinderRefinementInZ::type
      initial_cylinder_refinement_in_z_{};
  typename InitialSphereRadialGridPoints::type
      initial_sphere_radial_grid_points_{};
  typename InitialSphereThetaGridPoints::type
      initial_sphere_theta_grid_points_{};
  typename InitialSpherePhiGridPoints::type initial_sphere_phi_grid_points_{};
  typename InitialSphereRefinementInR::type initial_sphere_refinement_in_r_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      cylinder_lower_z_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      cylinder_upper_z_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      cylinder_mantle_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      sphere_inner_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      sphere_outer_boundary_condition_{};
  std::vector<std::string> block_names_{"Cylinder", "Sphere"};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{{"Base", {{"Cylinder"}}}, {"Globe", {{"Sphere"}}}};
  size_t num_blocks_{2};
};
}  // namespace domain::creators
