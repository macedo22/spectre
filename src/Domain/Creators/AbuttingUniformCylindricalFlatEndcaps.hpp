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
class UniformCylindricalFlatEndcap;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {
/*!
 * \brief Create a domain consisting of a two cylindrical flat endcap blocks
 * that abut each other like the MA and MB endcaps in
 * `::domain::creators::CylindricalBinaryCompactObject`.
 * Twice wraps `::domain::CoordinateMaps::UniformCylindricalFlatEndcap`.
 */
class AbuttingUniformCylindricalFlatEndcaps : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<::domain::CoordinateMap<
      Frame::BlockLogical, Frame::Inertial,
      ::domain::CoordinateMaps::ProductOf3Maps<
          ::domain::CoordinateMaps::Affine,
          ::domain::CoordinateMaps::Identity<1>,
          ::domain::CoordinateMaps::Interval>,
      ::domain::CoordinateMaps::ProductOf2Maps<
          ::domain::CoordinateMaps::PolarToCartesian,
          ::domain::CoordinateMaps::Identity<1>>,
      ::domain::CoordinateMaps::DiscreteRotation<3>,
      ::domain::CoordinateMaps::UniformCylindricalFlatEndcap,
      ::domain::CoordinateMaps::DiscreteRotation<3>>>;

  /*!
   * \brief Center of the imaginary sphere that would abut the lower surface of
   * the first cylindrical endcap
   */
  struct SphereCenterA {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center of the imaginary sphere that would abut the lower surface of "
        "the first cylindrical endcap. Cylinder A should be thought of as "
        "having the upside-down orientation of cylinder B, so the center of "
        "sphere A should instead be at a higher z coordinate than ZPlaneA. "
        "See domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Center of the imaginary sphere that would abut the lower surface of
   * the second cylindrical endcap
   */
  struct SphereCenterB {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center of the imaginary sphere that would abut the lower surface of "
        "the second cylindrical endcap. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Radius of the outer edge of the imaginary sphere that would abut
   * the lower surface of the first cylindrical endcap.
   */
  struct SphereRadiusA {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the imaginary sphere that would abut the lower surface of "
        "the first cylindrical endcap. A greater value will amount to a "
        "greater radius of curvature and thus a less curved lower surface for "
        "the cylindrical endcap. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Radius of the outer edge of the imaginary sphere that would abut
   * the lower surface of the second cylindrical endcap.
   */
  struct SphereRadiusB {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the imaginary sphere that would abut the lower surface of "
        "the second cylindrical endcap. A greater value will amount to a "
        "greater radius of curvature and thus a less curved lower surface for "
        "the cylindrical endcap. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Center of the cylinders' shared outer edge
   */
  struct CylinderCenter {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center of the shared upper surface of the cylindrical endcaps. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Radius of the cylinders' shared outer edge
   */
  struct CylinderRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the cylinders."};
  };

  /*!
   * \brief Plane of intersection between imaginary sphere and first
   * cylindrical endcap. See
   * `domain::CoordinateMaps::UniformCylindricalFlatEndcap` for more details.
   */
  struct ZPlaneA {
    using type = double;
    static constexpr Options::String help = {
        "Plane of intersection between imaginary sphere and first cylindrical "
        "endcap. Cylinder A should be thought of as having the upside-down "
        "orientation of cylinder B, so the center of sphere A should instead "
        "be at a higher z coordinate than ZPlaneA. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Plane of intersection between imaginary sphere and second
   * cylindrical endcap. See
   * `domain::CoordinateMaps::UniformCylindricalFlatEndcap` for more details.
   */
  struct ZPlaneB {
    using type = double;
    static constexpr Options::String help = {
        "Plane of intersection between imaginary sphere and second cylindrical "
        "endcap. See domain::CoordinateMaps::UniformCylindricalFlatEndcap for "
        "more details."};
  };

  /*!
   * \brief Direction of the normal to the flat cylindrical face of the second
   * endcap
   */
  struct RotationDirection {
    using type = int;
    static constexpr Options::String help = {
        "Direction of the normal to the flat cylindrical face of the second "
        "endcap: +x: +1, +y: +2, +z: +3, -x: -1, -y: -2, -z: -3. For now, the "
        "only supported direction is +x, so you must supply the value 1."};
  };

  /*!
   * \brief Whether or not to have nodal alignment on the shared face between
   * the two cylinders. This also determines whether or not the two cylinders
   * will be conforming block neighbors when `AreNeighbors == true`.
   */
  struct AlignNodesOnSharedFace {
    using type = bool;
    static constexpr Options::String help = {
        "Whether or not to have nodal alignment on the shared face between the "
        "two cylinders. This also determines whether or not the two cylinders "
        "are conforming block neighbors."};
  };

  /*!
   * \brief Whether or not the two cylinders are designated as block neighbors
   * or two standalone blocks.
   */
  struct AreNeighbors {
    using type = bool;
    static constexpr Options::String help = {
        "Whether or not the two cylinders are designated as block neighbors "
        "or two standalone blocks."};
  };

  /*!
   * \brief Initial number of radial gridpoints for the first filled cylinder.
   */
  struct InitialRadialGridPointsA {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for the first endcap."};
  };

  /*!
   * \brief Initial number of radial gridpoints for the second filled cylinder.
   */
  struct InitialRadialGridPointsB {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for the second endcap."};
  };

  /*!
   * \brief Initial number of \f$\theta\f$ gridpoints for first filled cylinder.
   * This is enforced to be odd for numerical stability.
   */
  struct InitialThetaGridPointsA {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in theta for the first endcap. It must "
        "be an odd number for stability. "
        "It must be <= 4 * (initial radial grid points) - 3."};
  };

  /*!
   * \brief Initial number of \f$\theta\f$ gridpoints for second filled
   * cylinder. This is enforced to be odd for numerical stability.
   */
  struct InitialThetaGridPointsB {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in theta for the second endcap. It must "
        "be an odd number for stability. "
        "It must be <= 4 * (initial radial grid points) - 3."};
  };

  /*!
   * \brief Initial number of z gridpoints for the first cylinder
   */
  struct InitialZGridPointsA {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in z for the first cylinder."};
  };

  /*!
   * \brief Initial number of z gridpoints for the second cylinder
   */
  struct InitialZGridPointsB {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in z for the second cylinder."};
  };

  /*!
   * \brief Initial refinement level in the z direction for the first cylinder
   */
  struct InitialRefinementInZA {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in the z-direction for the first cylinder."};
  };

  /*!
   * \brief Initial refinement level in the z direction for the second cylinder
   */
  struct InitialRefinementInZB {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in the z-direction for the second cylinder."};
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
  struct LowerZBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "LowerZ"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the lower spherical base of "
        "the cylinders.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Boundary condition on the radial boundary
   */
  template <typename BoundaryConditionsBase>
  struct MantleBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "Mantle"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the mantle of the cylinders, "
        "i.e. at the `CylinderRadius` in the radial direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options =
      tmpl::list<SphereCenterA, SphereCenterB, SphereRadiusA, SphereRadiusB,
                 CylinderCenter, CylinderRadius, ZPlaneA, ZPlaneB,
                 RotationDirection, AlignNodesOnSharedFace, AreNeighbors,
                 InitialRadialGridPointsA, InitialRadialGridPointsB,
                 InitialThetaGridPointsA, InitialThetaGridPointsB,
                 InitialZGridPointsA, InitialZGridPointsB,
                 InitialRefinementInZA, InitialRefinementInZB>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::append<
          basic_options,
          tmpl::list<
              LowerZBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              MantleBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>>>,
      basic_options>;

  static constexpr Options::String help{
      "Creates two abutting `UniformCylindricalFlatEndcap`s like the MA and MB "
      "endcaps in CylindricalBinaryCompactObject. When oriented along the "
      "z direction, Cylinder B's flat face points in +z like "
      "`::domain::CoordinateMaps::UniformCylindricalFlatEndcap`, while "
      "Cylinder A has the upside-down orientation."};

  AbuttingUniformCylindricalFlatEndcaps(
      typename SphereCenterA::type sphere_center_a,
      typename SphereCenterB::type sphere_center_b,
      typename SphereRadiusA::type sphere_radius_a,
      typename SphereRadiusB::type sphere_radius_b,
      typename CylinderCenter::type cylinder_center,
      typename CylinderRadius::type cylinder_radius,
      typename ZPlaneA::type z_plane_a, typename ZPlaneB::type z_plane_b,
      typename RotationDirection::type direction,
      typename AlignNodesOnSharedFace::type aligned_nodes_on_shared_face,
      typename AreNeighbors::type are_neighbors,
      typename InitialRadialGridPointsA::type initial_radial_grid_points_a,
      typename InitialRadialGridPointsB::type initial_radial_grid_points_b,
      typename InitialThetaGridPointsA::type initial_theta_grid_points_a,
      typename InitialThetaGridPointsB::type initial_theta_grid_points_b,
      typename InitialZGridPointsA::type initial_z_grid_points_a,
      typename InitialZGridPointsB::type initial_z_grid_points_b,
      typename InitialRefinementInZA::type initial_refinement_in_z_a,
      typename InitialRefinementInZB::type initial_refinement_in_z_b,
      const Options::Context& context = {});

  AbuttingUniformCylindricalFlatEndcaps(
      typename SphereCenterA::type sphere_center_a,
      typename SphereCenterB::type sphere_center_b,
      typename SphereRadiusA::type sphere_radius_a,
      typename SphereRadiusB::type sphere_radius_b,
      typename CylinderCenter::type cylinder_center,
      typename CylinderRadius::type cylinder_radius,
      typename ZPlaneA::type z_plane_a, typename ZPlaneB::type z_plane_b,
      typename RotationDirection::type direction,
      typename AlignNodesOnSharedFace::type aligned_nodes_on_shared_face,
      typename AreNeighbors::type are_neighbors,
      typename InitialRadialGridPointsA::type initial_radial_grid_points_a,
      typename InitialRadialGridPointsB::type initial_radial_grid_points_b,
      typename InitialThetaGridPointsA::type initial_theta_grid_points_a,
      typename InitialThetaGridPointsB::type initial_theta_grid_points_b,
      typename InitialZGridPointsA::type initial_z_grid_points_a,
      typename InitialZGridPointsB::type initial_z_grid_points_b,
      typename InitialRefinementInZA::type initial_refinement_in_z_a,
      typename InitialRefinementInZB::type initial_refinement_in_z_b,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          lower_z_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          mantle_boundary_condition = nullptr,
      const Options::Context& context = {});

  AbuttingUniformCylindricalFlatEndcaps() = default;
  AbuttingUniformCylindricalFlatEndcaps(
      const AbuttingUniformCylindricalFlatEndcaps&) = delete;
  AbuttingUniformCylindricalFlatEndcaps(
      AbuttingUniformCylindricalFlatEndcaps&&) = default;
  AbuttingUniformCylindricalFlatEndcaps& operator=(
      const AbuttingUniformCylindricalFlatEndcaps&) = delete;
  AbuttingUniformCylindricalFlatEndcaps& operator=(
      AbuttingUniformCylindricalFlatEndcaps&&) = default;
  ~AbuttingUniformCylindricalFlatEndcaps() override = default;

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
  typename SphereCenterA::type sphere_center_a_{};
  typename SphereCenterB::type sphere_center_b_{};
  typename SphereRadiusA::type sphere_radius_a_{};
  typename SphereRadiusB::type sphere_radius_b_{};
  typename CylinderCenter::type cylinder_center_{};
  typename CylinderRadius::type cylinder_radius_{};
  typename ZPlaneA::type z_plane_a_{};
  typename ZPlaneB::type z_plane_b_{};
  typename RotationDirection::type direction_{};
  typename AlignNodesOnSharedFace::type aligned_nodes_on_shared_face_{};
  typename AreNeighbors::type are_neighbors_{};
  typename InitialRadialGridPointsA::type initial_radial_grid_points_a_{};
  typename InitialRadialGridPointsB::type initial_radial_grid_points_b_{};
  typename InitialThetaGridPointsA::type initial_theta_grid_points_a_{};
  typename InitialThetaGridPointsB::type initial_theta_grid_points_b_{};
  typename InitialZGridPointsA::type initial_z_grid_points_a_{};
  typename InitialZGridPointsB::type initial_z_grid_points_b_{};
  size_t initial_refinement_in_z_a_{};
  size_t initial_refinement_in_z_b_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      lower_z_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      mantle_boundary_condition_{};
  std::vector<std::string> block_names_{"FlatEndcapA", "FlatEndcapB"};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{{"FlatEndcaps", {{"FlatEndcapA", "FlatEndcapB"}}}};
  size_t num_blocks_{2};
};
}  // namespace domain::creators
