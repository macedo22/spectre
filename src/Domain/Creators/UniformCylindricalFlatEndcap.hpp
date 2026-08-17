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
 * \brief Create a domain consisting of a single cylindrical flat endcap block.
 * Wraps `::domain::CoordinateMaps::UniformCylindricalFlatEndcap`.
 */
class UniformCylindricalFlatEndcap : public DomainCreator<3> {
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
      ::domain::CoordinateMaps::UniformCylindricalFlatEndcap,
      ::domain::CoordinateMaps::DiscreteRotation<3>>>;

  /*!
   * \brief Center of the imaginary sphere that would abut the lower surface of
   * the cylindrical endcap
   */
  struct SphereCenter {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center of the imaginary sphere that would abut the lower surface of "
        "the cylindrical endcap. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Radius of the outer edge of the imaginary sphere that would abut the
   * lower surface of the cylindrical endcap.
   */
  struct SphereRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the imaginary sphere that would abut the lower surface of "
        "the cylindrical endcap. A greater value will amount to a greater "
        "radius of curvature and thus a less curved lower surface for the "
        "cylindrical endcap. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Center of the cylinder's outer edge
   */
  struct CylinderCenter {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center of the upper surface of the cylindrical endcap. See "
        "domain::CoordinateMaps::UniformCylindricalFlatEndcap for more "
        "details."};
  };

  /*!
   * \brief Radius of the cylinder's outer edge
   */
  struct CylinderRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the cylinder."};
  };

  /*!
   * \brief Plane of intersection between imaginary sphere and cylindrical
   * endcap. See `domain::CoordinateMaps::UniformCylindricalFlatEndcap` for more
   * details.
   */
  struct ZPlane {
    using type = double;
    static constexpr Options::String help = {
        "Plane of intersection between imaginary sphere and cylindrical "
        "endcap. See domain::CoordinateMaps::UniformCylindricalFlatEndcap for "
        "more details."};
  };

  /*!
   * \brief Direction of the normal to the flat cylindrical face of the endcap
   */
  struct RotationDirection {
    using type = int;
    static constexpr Options::String help = {
        "Direction of the normal to the flat cylindrical face of the endcap: "
        "+x: +1, +y: +2, +z: +3, -x: -1, -y: -2, -z: -3. Specify 3 to keep "
        "the endcap oriented the same (+z) as "
        "`domain::CoordinateMaps::UniformCylindricalFlatEndcap`."};
  };

  /*!
   * \brief Initial number of radial gridpoints for filled cylinder.
   */
  struct InitialRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points."};
  };

  /*!
   * \brief Initial number of \f$\theta\f$ gridpoints for filled cylinder. This
   * is enforced to be odd for numerical stability.
   */
  struct InitialThetaGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in theta. It must be an odd number for "
        "stability. It must be <= 4 * (initial radial grid points) - 3."};
  };

  /*!
   * \brief Initial number of z gridpoints for the cylinder
   */
  struct InitialZGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in z."};
  };

  /*!
   * \brief Initial refinement levels in the z direction
   */
  struct InitialRefinementInZ {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in the z-direction."};
  };

  /*!
   * \brief Boundary conditions group
   */
  struct BoundaryConditions {
    static constexpr Options::String help =
        "Options for the boundary conditions";
  };

  /*!
   * \brief Boundary condition on the lower base
   */
  template <typename BoundaryConditionsBase>
  struct LowerZBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "LowerZ"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the lower spherical base of "
        "the cylinder.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Boundary condition on the upper base
   */
  template <typename BoundaryConditionsBase>
  struct UpperZBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "UpperZ"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the upper flat base of the "
        "cylinder.";
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
        "The boundary condition to be imposed on the mantle of the cylinder, "
        "i.e. at the `CylinderRadius` in the radial direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options =
      tmpl::list<SphereCenter, SphereRadius, CylinderCenter, CylinderRadius,
                 ZPlane, RotationDirection, InitialRadialGridPoints,
                 InitialThetaGridPoints, InitialZGridPoints,
                 InitialRefinementInZ>;

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
              UpperZBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              MantleBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>>>,
      basic_options>;

  static constexpr Options::String help{
      "Creates a single UniformCylindricalFlatEndcap."};

  UniformCylindricalFlatEndcap(
      typename SphereCenter::type sphere_center,
      typename SphereRadius::type sphere_radius,
      typename CylinderCenter::type cylinder_center,
      typename CylinderRadius::type cylinder_radius,
      typename ZPlane::type z_plane, typename RotationDirection::type direction,
      typename InitialRadialGridPoints::type initial_radial_grid_points,
      typename InitialThetaGridPoints::type initial_theta_grid_points,
      typename InitialZGridPoints::type initial_z_grid_points,
      typename InitialRefinementInZ::type initial_refinement_in_z,
      const Options::Context& context = {});

  UniformCylindricalFlatEndcap(
      typename SphereCenter::type sphere_center,
      typename SphereRadius::type sphere_radius,
      typename CylinderCenter::type cylinder_center,
      typename CylinderRadius::type cylinder_radius,
      typename ZPlane::type z_plane, typename RotationDirection::type direction,
      typename InitialRadialGridPoints::type initial_radial_grid_points,
      typename InitialThetaGridPoints::type initial_theta_grid_points,
      typename InitialZGridPoints::type initial_z_grid_points,
      typename InitialRefinementInZ::type initial_refinement_in_z,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          lower_z_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          upper_z_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          mantle_boundary_condition = nullptr,
      const Options::Context& context = {});

  UniformCylindricalFlatEndcap() = default;
  UniformCylindricalFlatEndcap(const UniformCylindricalFlatEndcap&) = delete;
  UniformCylindricalFlatEndcap(UniformCylindricalFlatEndcap&&) = default;
  UniformCylindricalFlatEndcap& operator=(const UniformCylindricalFlatEndcap&) =
      delete;
  UniformCylindricalFlatEndcap& operator=(UniformCylindricalFlatEndcap&&) =
      default;
  ~UniformCylindricalFlatEndcap() override = default;

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
  typename SphereRadius::type sphere_radius_{};
  typename CylinderCenter::type cylinder_center_{};
  typename CylinderRadius::type cylinder_radius_{};
  typename ZPlane::type z_plane_{};
  typename RotationDirection::type direction_{};
  typename InitialRadialGridPoints::type initial_radial_grid_points_{};
  typename InitialThetaGridPoints::type initial_theta_grid_points_{};
  typename InitialZGridPoints::type initial_z_grid_points_{};
  size_t initial_refinement_in_z_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      lower_z_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      upper_z_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      mantle_boundary_condition_{};
  std::vector<std::string> block_names_{{"FlatEndcap"}};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{{"FlatEndcap", {{"FlatEndcap"}}}};
  size_t num_blocks_{1};
};
}  // namespace domain::creators
