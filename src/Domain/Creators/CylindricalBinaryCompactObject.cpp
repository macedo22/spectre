// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"

#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

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
#include "Domain/CoordinateMaps/UniformCylindricalEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
std::array<double, 3> rotate_to_z_axis(const std::array<double, 3> input) {
  return discrete_rotation(
      OrientationMap<3>{std::array<Direction<3>, 3>{Direction<3>::lower_zeta(),
                                                    Direction<3>::upper_eta(),
                                                    Direction<3>::upper_xi()}},
      input);
}
std::array<double, 3> rotate_from_z_to_x_axis(
    const std::array<double, 3> input) {
  return discrete_rotation(
      OrientationMap<3>{std::array<Direction<3>, 3>{Direction<3>::upper_zeta(),
                                                    Direction<3>::upper_eta(),
                                                    Direction<3>::lower_xi()}},
      input);
}
std::array<double, 3> flip_about_xy_plane(const std::array<double, 3> input) {
  return std::array<double, 3>{input[0], input[1], -input[2]};
}
}  // namespace

namespace domain::creators {
CylindricalBinaryCompactObject::CylindricalBinaryCompactObject(
    std::array<double, 3> center_A, std::array<double, 3> center_B,
    double radius_A, double radius_B, bool include_inner_sphere_A,
    bool include_inner_sphere_B, double outer_radius,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_grid_points,
    std::optional<bco::TimeDependentMapOptions<true>> time_dependent_options,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : center_A_(rotate_to_z_axis(center_A)),
      center_B_(rotate_to_z_axis(center_B)),
      radius_A_(radius_A),
      radius_B_(radius_B),
      include_inner_sphere_A_(include_inner_sphere_A),
      include_inner_sphere_B_(include_inner_sphere_B),
      outer_radius_(outer_radius),
      inner_boundary_condition_(std::move(inner_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      time_dependent_options_(std::move(time_dependent_options)) {
  if (center_A_[2] <= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of the input CenterA is expected to be positive");
  }
  if (center_B_[2] >= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of the input CenterB is expected to be negative");
  }
  if (radius_A_ <= 0.0 or radius_B_ <= 0.0) {
    PARSE_ERROR(context, "RadiusA and RadiusB are expected to be positive");
  }
  if (radius_A_ < radius_B_) {
    PARSE_ERROR(context, "RadiusA should not be smaller than RadiusB");
  }
  if (std::abs(center_A_[2]) > std::abs(center_B_[2])) {
    PARSE_ERROR(context,
                "We expect |x_A| <= |x_B|, for x the x-coordinate of either "
                "CenterA or CenterB.  We should roughly have "
                "RadiusA x_A + RadiusB x_B = 0 (i.e. for BBHs the "
                "center of mass should be about at the origin).");
  }
  // The value 3.0 * (center_A_[2] - center_B_[2]) is what is
  // chosen in SpEC as the inner radius of the innermost outer sphere.
  if (outer_radius_ < 3.0 * (center_A_[2] - center_B_[2])) {
    PARSE_ERROR(context,
                "OuterRadius is too small. Please increase it "
                "beyond "
                    << 3.0 * (center_A_[2] - center_B_[2]));
  }

  if ((outer_boundary_condition_ == nullptr) xor
      (inner_boundary_condition_ == nullptr)) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(inner_boundary_condition_) or
      is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a binary domain");
  }

  // The choices made below for the quantities xi, z_cutting_plane_,
  // and xi_min_sphere_e are the ones made in SpEC, and in the
  // Appendix of https://arxiv.org/abs/1206.3015.  Other choices could
  // be made that would still result in a reasonable Domain. In
  // particular, during a SpEC BBH evolution the excision boundaries
  // can sometimes get too close to z_cutting_plane_, and the
  // simulation must be halted and regridded with a different choice
  // of z_cutting_plane_, so it may be possible to choose a different
  // initial value of z_cutting_plane_ that reduces the number of such
  // regrids or eliminates them.

  // xi is the quantity in Eq. (A10) of
  // https://arxiv.org/abs/1206.3015 that represents how close the
  // cutting plane is to either center.  Unfortunately, there is a
  // discrepancy between what xi means in the paper and what it is in
  // the code.  I (Mark) think that this is a typo in the paper,
  // because otherwise the domain doesn't make sense.  To fix this,
  // either Eq. (A9) in the paper should have xi -> 1-xi, or Eq. (A10)
  // should have x_A and x_B swapped.
  // Here we will use the same definition of xi in Eq. (A10), but we
  // will swap xi -> 1-xi in Eq. (A9).
  // Therefore, xi = 0 means that the cutting plane passes through the center of
  // object B, and xi = 1 means that the cutting plane passes through
  // the center of object A.  Note that for |x_A| <= |x_B| (as assumed
  // above), xi is always <= 1/2.
  constexpr double xi_min = 0.25;
  // Same as Eq. (A10)
  const double xi =
      std::max(xi_min, std::abs(center_A_[2]) /
                           (std::abs(center_A_[2]) + std::abs(center_B_[2])));

  // Compute cutting plane
  // This is Eq. (A9) with xi -> 1-xi.
  z_cutting_plane_ = cut_spheres_offset_factor_ *
                     ((1.0 - xi) * center_B_[2] + xi * center_A_[2]);

  // Add SphereE blocks if necessary.  Note that
  // https://arxiv.org/abs/1206.3015 has a mistake just above
  // Eq. (A.11) and the same mistake above Eq. (A.20), where it lists
  // the wrong mass ratio (for BBHs). The correct statement is that if
  // xi <= 1/3, this means that the mass ratio (for BBH) is large (>=2)
  // and we should add SphereE blocks.
  constexpr double xi_min_sphere_e = 1.0 / 3.0;
  if (xi <= xi_min_sphere_e) {
    // The following ERROR will be removed in an upcoming PR that
    // will support higher mass ratios.
    ERROR(
        "We currently only support domains where objects A and B are "
        "approximately the same size, and approximately the same distance from "
        "the origin.  More technically, we support xi > "
        << xi_min_sphere_e << ", but the value of xi is " << xi
        << ". Support for more general domains will be added in the near "
           "future");
  }

  // Create grid anchors in x direction from unrotated input centers
  grid_anchors_ = bco::create_grid_anchors(center_A, center_B);

  // Build the set of cylindrical block groups and block names so the
  // validation below can distinguish spherical-harmonic blocks from other
  // blocks and block groups.
  std::unordered_set<std::string> filled_cylinder_names{};
  std::unordered_set<std::string> hollow_cylinder_names{};

  // Create cylinder block names and groups
  auto add_filled_cylinder_name = [this, &filled_cylinder_names](
                                      const std::string& prefix,
                                      const std::string& group_name) {
    const std::string name = std::string(prefix).append("FilledCylinder");
    block_names_.push_back(name);
    block_groups_[group_name].insert(name);
    block_positions_[name] = block_names_.size() - 1;
    filled_cylinder_names.insert(name);
    filled_cylinder_names.insert(group_name);
  };
  auto add_cylinder_name = [this, &hollow_cylinder_names](
                               const std::string& prefix,
                               const std::string& group_name) {
    const std::string name = std::string(prefix).append("Cylinder");
    block_names_.push_back(name);
    block_groups_[group_name].insert(name);
    block_positions_[name] = block_names_.size() - 1;
    hollow_cylinder_names.insert(name);
    hollow_cylinder_names.insert(group_name);
  };

  // CA Filled Cylinder
  add_filled_cylinder_name("CA", "Outer");

  // CA Cylinder
  add_cylinder_name("CA", "Outer");

  // EA Filled Cylinder
  add_filled_cylinder_name("EA", "InnerA");

  // EA Cylinder
  add_cylinder_name("EA", "InnerA");

  // EB Filled Cylinder
  add_filled_cylinder_name("EB", "InnerB");

  // EB Cylinder
  add_cylinder_name("EB", "InnerB");

  // MA Filled Cylinder
  add_filled_cylinder_name("MA", "InnerA");

  // MB Filled Cylinder
  add_filled_cylinder_name("MB", "InnerB");

  // CB Filled Cylinder
  add_filled_cylinder_name("CB", "Outer");

  // CB Cylinder
  add_cylinder_name("CB", "Outer");

  // combine filled and hollow cylinder blocks and groups into one set
  std::unordered_set<std::string> all_cylinder_names;
  std::set_union(
      std::begin(filled_cylinder_names), std::end(filled_cylinder_names),
      std::begin(hollow_cylinder_names), std::end(hollow_cylinder_names),
      std::inserter(all_cylinder_names, std::begin(all_cylinder_names)));

  // Build the set of spherical-harmonic shell block groups and block names so
  // the validation below can distinguish spherical-harmonic blocks from other
  // blocks and block groups.
  std::unordered_set<std::string> spherical_harmonic_shell_names{};

  // Create block names and groups
  auto add_spherical_shell_name = [this, &spherical_harmonic_shell_names](
                                      const std::string& prefix,
                                      const std::string& group_name,
                                      const size_t shell_number) {
    const std::string name = std::string(prefix).append("Shell").append(
        std::to_string(shell_number));
    block_names_.push_back(name);
    block_groups_[group_name].insert(name);
    block_positions_[name] = block_names_.size() - 1;
    spherical_harmonic_shell_names.insert(name);
    spherical_harmonic_shell_names.insert(group_name);
  };

  auto add_inner_spherical_shells_for_sphere =
      [this, &add_spherical_shell_name](
          const std::string& prefix, const std::string& group_name,
          const std::array<double, 3>& center_rotated_to_z_axis,
          const double inner_radius) {
        // In spec/InputFiles/bbh/DoMulitpleRuns.input, it sets
        //   RelativeDelta = 0.40
        // which it says is set by
        //   RelativeDeltaR := DeltaR/Ravg.
        // In spec/InputFiles/bbh/GrDomain.input, it then sets
        //   RelativeDeltaR=1.1*__RelativeDeltaR__
        // for SphereA and SphereB.
        const double relative_delta_r = 0.44;
        // In spec/Dust/Domain/Subdomain/CreateTouchingCutSphereWedgesBBH.cpp,
        //   rA = fabs(Geometry().CutX()-mCenterA[0])
        // and
        //   rB = fabs(Geometry().CutX()-mCenterB[0])
        // where Geometry().CutX() is the x coordinate of the cutting plane.
        const double distance_to_cutting_plane =
            fabs(z_cutting_plane_ - gsl::at(center_rotated_to_z_axis, 2));
        // The below implements SpEC logic for determining an inner sphere's
        // outer radius, how many shells it should have, and its radial
        // partitioning when specific radial partitioning is not specified by
        // user input. Since radial partitioning is not currently an input
        // option for this domain, it will always be automatically determined in
        // this way.
        //
        // In spec/Dust/Domain/Subdomain/CreateTouchingSphericalShellHelper.cpp,
        // NextRadiusOutsideRmax is distance_to_cutting_plane. See spec branch
        // if(NextRadiusOutsideRmax>0 and not RMaxIsDefined), which is what is
        // implemented below.
        const double distance_to_cutting_plane_over_inner_radius =
            distance_to_cutting_plane / inner_radius;
        int num_shells =
            std::round(std::log(distance_to_cutting_plane_over_inner_radius) /
                       std::log(1.0 + relative_delta_r)) -
            1;
        // Note: The SpEC logic then does:
        //   if(mNShells==0) mNShells++;
        //   if(mNShells<0) mNShells=0;
        // This domain, however, currently accepts a boolean for whether or not
        // to include inner spheres. It's possible that a user would specify
        // `true` to ask for an inner sphere but then num_shells above ends up
        // being negative. By the SpEC logic, this would set num_shells = 0, but
        // this would directly go against the user's request to include an
        // inner sphere. For this reason, we instead choose to set
        // num_shells = 1 when this happens so as to still honor the user's
        // request to have an inner sphere at all. This can later be updated to
        // match SpEC's behavior if this domain's input options to include or
        // exclude an inner sphere are removed.
        if (num_shells <= 0) {
          num_shells = 1;
        }

        const double coef = pow(distance_to_cutting_plane_over_inner_radius,
                                1.0 / static_cast<double>(num_shells + 1));
        // std::vector<double> radii{};
        // radii.reserve(num_shells + 1);
        // mRadii.assign(MV::Size(mNShells + 1), rmin);
        // for (int k = 1; k <= mNShells; ++k)
        //   mRadii[k] = mRadii[k - 1] * coef;
        double outermost_radius = inner_radius;
        for (size_t shell_number = 0;
             shell_number < static_cast<size_t>(num_shells); shell_number++) {
          add_spherical_shell_name(prefix, group_name, shell_number);
          outermost_radius *= coef;
        }

        return outermost_radius;
      };

  outer_radius_A_ = add_inner_spherical_shells_for_sphere(
      "InnerA", "InnerSphereA", center_A_, radius_A_);
  outer_radius_B_ = add_inner_spherical_shells_for_sphere(
      "InnerB", "InnerSphereB", center_B_, radius_B_);
  add_spherical_shell_name("Outer", "OuterSphere", 0);

  number_of_blocks_ = block_names_.size();
  ASSERT(number_of_blocks_ == block_positions_.size(),
         "Size of block_positions_ map should be equal to the number of blocks "
         "in the domain.");

  // Since BinaryCompactObject::InitialGridPoints type differs from
  // CylindricalBinaryCompactObject::InitialGridPoints type, need to first
  // create the BCO-compatible type with the CBCO data to be able to reuse the
  // functionality of bco::validate_initial_grid_points() and
  // bco::set_initial_grid_points().
  const auto bco_initial_grid_points = std::visit(
      [](const auto& value) {
        return BinaryCompactObject::InitialGridPoints::type{value};
      },
      initial_grid_points);
  // Validate that the input file has the correct format for
  // InitialGridPoints. No need to validate the format for InitialRefinement
  // because it does not accept a map of strings to possibly
  // differently-sized arrays for refinement. If a map is provided, it already
  // only accepts a map of size_t keys.
  bco::validate_initial_grid_points(context, bco_initial_grid_points,
                                    spherical_harmonic_shell_names,
                                    filled_cylinder_names);

  // For expanding initial refinement and grid points over all blocks
  const ExpandOverBlocks<std::array<size_t, 3>> expand_over_blocks{
      block_names_, block_groups_};
  try {
    // Since BinaryCompactObject::InitialRefinement map type differs from
    // CylindricalBinaryCompactObject::InitialRefinement map type, need to first
    // create the BCO-compatible type with the CBCO data to be able to reuse the
    // functionality of bco::set_initial_refinement().
    using bco_ref_map_type =
        std::unordered_map<std::string,
                           std::variant<std::array<size_t, 3>, size_t>>;
    using cbco_ref_map_type = std::unordered_map<std::string, size_t>;
    const auto bco_initial_refinement =
        std::holds_alternative<size_t>(initial_refinement)
            ? BinaryCompactObject::InitialRefinement::type{std::get<size_t>(
                  initial_refinement)}
            : BinaryCompactObject::InitialRefinement::type{bco_ref_map_type{
                  std::get<cbco_ref_map_type>(initial_refinement).begin(),
                  std::get<cbco_ref_map_type>(initial_refinement).end()}};
    initial_refinement_ = bco::set_initial_refinement(
        expand_over_blocks, bco_initial_refinement,
        spherical_harmonic_shell_names, all_cylinder_names);
    // If a global single-number h-refinement was used, post-process the
    // expanded cylinder and spherical shell blocks to make the angular
    // directions have h refinement = 0.
    if (std::holds_alternative<size_t>(initial_refinement)) {
      for (const auto& [name, position] : block_positions_) {
        if (name.find("Cylinder") != std::string::npos) {
          // Set cylinder h refinement to {0, 0, z}
          initial_refinement_[position][0] = 0;
          initial_refinement_[position][1] = 0;
        } else if (name.find("Shell") != std::string::npos) {
          // Set spherical shell h refinement to {r, 0, 0}
          initial_refinement_[position][1] = 0;
          initial_refinement_[position][2] = 0;
        }
      }
    }
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialRefinement': " << error.what());
  }

  // Validate angular h-refinement == 0 in cylinder and spherical shell blocks
  for (const auto& [name, position] : block_positions_) {
    if (name.find("Cylinder") != std::string::npos) {
      if (gsl::at(gsl::at(initial_refinement_, position), 0) != 0 or
          gsl::at(gsl::at(initial_refinement_, position), 1) != 0) {
        PARSE_ERROR(context,
                    "Angular h-refinement is not supported for cylindrical "
                    "blocks. Specify refinement for "
                        << name << " as a single number.");
      }
    } else if (name.find("Shell") != std::string::npos) {
      if (gsl::at(gsl::at(initial_refinement_, position), 1) != 0 or
          gsl::at(gsl::at(initial_refinement_, position), 2) != 0) {
        PARSE_ERROR(context,
                    "Angular h-refinement is not supported for "
                    "spherical-harmonic shell blocks. Specify refinement for "
                        << name << " as a single number.");
      }
    }
  }

  try {
    initial_grid_points_ = bco::set_initial_grid_points(
        expand_over_blocks, bco_initial_grid_points,
        spherical_harmonic_shell_names, filled_cylinder_names);
    // If a global single-number p-refinement was used, post-process the
    // expanded filled cylinder blocks to make the angular directions have the
    // correct number of spectral points for ZernikeB2.
    if (std::holds_alternative<size_t>(initial_grid_points)) {
      for (const auto& [name, position] : block_positions_) {
        if (name.find("FilledCylinder") != std::string::npos) {
          // note for ZernikeB2:
          // radial_points = (theta_modes / 2) + 1 + (theta_modes % 2), so one
          // could have odd or even theta_modes for the same radial_points.
          // Choosing the even theta_modes for the same radial_points means:
          //   theta_modes = 2 * (radial_points - 1)
          //   theta_points = 2 * theta_modes + 1 = 4 * radial_modes - 3
          // Choosing the odd theta_modes for the same radial_points means:
          //   theta_modes = 2 * (radial_points - 2) + 1
          //   theta_points = 2 * theta_modes + 1 = 4 * radial_modes - 5
          // Here, we choose the even case to get one extra theta_mode out of
          // radial_points.
          initial_grid_points_[position][1] =
              4 * gsl::at(gsl::at(initial_grid_points_, position), 0) - 3;
        }
      }
    }
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialGridPoints': " << error.what());
  }

  // Validate p-refinement values in cylinder and spherical shell blocks
  for (const auto& [name, position] : block_positions_) {
    if (name.find("FilledCylinder") != std::string::npos) {
      // Validate number of radial points for filled cylinders is > 2
      if (gsl::at(gsl::at(initial_grid_points_, position), 0) <= 2) {
        PARSE_ERROR(context,
                    "Filled cylindrical block "
                        << name
                        << " must have more than 2 radial grid points.");
      }

      // Validate number of angular grid points in filled cylinder blocks is
      // what is expected by ZernikeB2. The Zernike disk is fully specified by
      // either the number of radial points or the number of theta points, so
      // check that they relate as expected.
      const size_t num_theta_modes =
          gsl::at(gsl::at(initial_grid_points_, position), 1) / 2;
      const size_t expected_num_r_points =
          (num_theta_modes / 2) + 1 + (num_theta_modes % 2);
      if (gsl::at(gsl::at(initial_grid_points_, position), 0) !=
          expected_num_r_points) {
        PARSE_ERROR(context,
                    "Filled cylinder blocks must have "
                    "num_r_points = ((num_theta_points / 2) / 2) + 1 + "
                    "((num_theta_points / 2) % 2). Specify grid points for "
                        << name << " as [num_radial_points, num_z_points].");
      }
    }
    if (name.find("Cylinder") != std::string::npos) {
      // Validate number of angular grid points in all cylindrical blocks are
      // odd.
      if (gsl::at(gsl::at(initial_grid_points_, position), 1) % 2 == 0) {
        PARSE_ERROR(context,
                    "Cylindrical block "
                        << name
                        << " must have an odd number of angular grid points.");
      }
    } else if (name.find("Shell") != std::string::npos) {
      // For spherical-harmonic shell blocks, initial_number_of_grid_points_
      // stores {n_radial, l_max, m_max}. First validate that l_max == m_max,
      // then convert (l_max, m_max) to the number of collocation points the
      // spherical-harmonic basis uses in each angular direction.
      const size_t l_max = gsl::at(gsl::at(initial_grid_points_, position), 1);
      const size_t m_max = gsl::at(gsl::at(initial_grid_points_, position), 2);
      if (l_max != m_max) {
        PARSE_ERROR(context,
                    "Spherical-harmonic shell blocks must have L_max = M_max. "
                    "Specify grid points for "
                        << name << " as [radial_points, L_max].");
      }
      initial_grid_points_[position][1] = ylm::Spherepack::n_theta_points(
          gsl::at(gsl::at(initial_grid_points_, position), 1));
      initial_grid_points_[position][2] = ylm::Spherepack::n_phi_points(
          gsl::at(gsl::at(initial_grid_points_, position), 2));
    }
  }

  // Build time-dependent maps
  // The size map, which is applied from the grid to distorted frame, currently
  // needs to start and stop at certain radii around each excision. If the inner
  // spheres aren't included, the outer radii would have to be in the middle of
  // a block. With the inner spheres, the outer radii can be at block
  // boundaries.
  if (time_dependent_options_.has_value() and
      not(include_inner_sphere_A and include_inner_sphere_B)) {
    PARSE_ERROR(context,
                "To use the CylindricalBBH domain with time-dependent maps, "
                "you must include the inner spheres for both objects. "
                "Currently, one or both objects is missing the inner sphere.");
  }

  if (time_dependent_options_.has_value()) {
    const double inner_common_radius = 3.0 * (center_A_[2] - center_B_[2]);
    const auto center_A_aligned = rotate_from_z_to_x_axis(center_A_);
    const auto center_B_aligned = rotate_from_z_to_x_axis(center_B_);
    time_dependent_options_->build_maps(
        std::array{center_A_aligned, center_B_aligned}, std::nullopt,
        std::nullopt,
        std::array{z_cutting_plane_,
                   0.5 * (center_A_aligned[1] + center_B_aligned[1]),
                   0.5 * (center_A_aligned[2] + center_B_aligned[2])},
        std::array{radius_A_, outer_radius_A_},
        std::array{radius_B_, outer_radius_B_}, false, false,
        inner_common_radius, outer_radius_);
  }
}

Domain<3> CylindricalBinaryCompactObject::create_domain() const {
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coordinate_maps{};

  const OrientationMap<3> rotate_to_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
      Direction<3>::lower_xi()}};

  const OrientationMap<3> rotate_to_minus_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
      Direction<3>::upper_xi()}};

  // 180 degree rotation about a cylinder's axis
  const OrientationMap<3> half_turn_about_zeta{std::array<Direction<3>, 3>{
      Direction<3>::lower_xi(), Direction<3>::lower_eta(),
      Direction<3>::upper_zeta()}};

  const OrientationMap<3> aligned = OrientationMap<3>::create_aligned();

  const std::array<double, 3> center_cutting_plane = {0.0, 0.0,
                                                      z_cutting_plane_};

  // The labels EA, EB, EE, etc are from Figure 20 of
  // https://arxiv.org/abs/1206.3015
  //
  // center_EA and radius_EA are the center and outer-radius of the
  // cylindered-sphere EA in Figure 20.
  //
  // center_EB and radius_EB are the center and outer-radius of the
  // cylindered-sphere EB in Figure 20.
  //
  // radius_MB is eq. A16 or A23 in the paper (depending on whether
  // the EE spheres exist), and is the radius of the circle where the EB
  // sphere intersects the cutting plane.
  const std::array<double, 3> center_EA = {
      0.0, 0.0, cut_spheres_offset_factor_ * center_A_[2]};
  const std::array<double, 3> center_EB = {
      0.0, 0.0, center_B_[2] * cut_spheres_offset_factor_};
  const double radius_MB =
      std::abs(cut_spheres_offset_factor_ * center_B_[2] - z_cutting_plane_);
  const double radius_EA =
      sqrt(square(center_EA[2] - z_cutting_plane_) + square(radius_MB));
  const double radius_EB =
      sqrt(2.0) * std::abs(center_EB[2] - z_cutting_plane_);

  // Construct a coordinate map that goes from logical coordinates to a unit
  // right cylinder block. The radii and bounds are what are expected by the
  // UniformCylindricalEndCap and UniformCylindricalFlatEndCap maps.
  const double cylinder_inner_radius = 0.0;
  const double cylinder_outer_radius = 1.0;
  const double cylinder_lower_bound_z = -1.0;
  const double cylinder_upper_bound_z = 1.0;

  const auto logical_to_cylinder_map =
      cyl_coordinate_map(cylinder_inner_radius, cylinder_outer_radius,
                         cylinder_lower_bound_z, cylinder_upper_bound_z);

  // Lambda that takes a pre-rotation map, a UniformCylindricalEndcap or a
  // UniformCylindricalFlatEndcap map and a DiscreteRotation map, composes it
  // with the logical-to-cylinder map, and adds it to the list of
  // coordinate maps. Also adds boundary conditions if requested. The
  // pre-rotation map is used by blocks at the cutting plane to achieve nodal
  // alignment with their block neighbor on the other side of the plane.
  auto add_endcap_to_list_of_maps =
      [&coordinate_maps, &logical_to_cylinder_map](
          const CoordinateMaps::DiscreteRotation<3>& pre_rotation_map,
          const auto& endcap_map,
          const CoordinateMaps::DiscreteRotation<3>& rotation_map) {
        auto new_logical_to_cylinder_map = ::domain::push_back(
            ::domain::push_back(
                ::domain::push_back(logical_to_cylinder_map, pre_rotation_map),
                endcap_map),
            rotation_map);

        coordinate_maps.emplace_back(
            std::make_unique<
                std::decay_t<decltype(new_logical_to_cylinder_map)>>(
                std::move(new_logical_to_cylinder_map)));
      };

  // Construct a coordinate map that goes from logical coordinates to a unit
  // right cylindrical shell block. The radii and bounds are what are expected
  // by the UniformCylindricalSide map.
  const double cylindrical_shell_inner_radius = 1.0;
  const double cylindrical_shell_outer_radius = 2.0;
  const double cylindrical_shell_lower_bound_z = -1.0;
  const double cylindrical_shell_upper_bound_z = 1.0;

  const auto logical_to_cylindrical_shell_map = cyl_coordinate_map(
      cylindrical_shell_inner_radius, cylindrical_shell_outer_radius,
      cylindrical_shell_lower_bound_z, cylindrical_shell_upper_bound_z);

  // Lambda that takes a pre-rotation map, a UniformCylindricalSide map, and a
  // DiscreteRotation map, composes it with the logical-to-cylinder maps, and
  // adds it to the list of coordinate maps.  Also adds boundary conditions if
  // requested.  The pre-rotation map is used by blocks at the cutting plane to
  // achieve nodal alignment with their block neighbor on the other side of the
  // plane.
  auto add_side_to_list_of_maps =
      [&coordinate_maps, &logical_to_cylindrical_shell_map](
          const CoordinateMaps::DiscreteRotation<3>& pre_rotation_map,
          const CoordinateMaps::UniformCylindricalSide& side_map,
          const CoordinateMaps::DiscreteRotation<3>& rotation_map) {
        auto new_logical_to_cylindrical_shell_map = ::domain::push_back(
            ::domain::push_back(
                ::domain::push_back(logical_to_cylindrical_shell_map,
                                    pre_rotation_map),
                side_map),
            rotation_map);

        coordinate_maps.emplace_back(
            std::make_unique<
                std::decay_t<decltype(new_logical_to_cylindrical_shell_map)>>(
                std::move(new_logical_to_cylindrical_shell_map)));
      };

  // Inner radius of the outer C shell.
  const double inner_radius_C = 3.0 * (center_A_[2] - center_B_[2]);

  // z_cut_CA_lower is the lower z_plane position for the CA endcap,
  // defined by https://arxiv.org/abs/1206.3015 in the bulleted list
  // after Eq. (A.19) EXCEPT that here we use a factor of 1.6 instead of 1.5
  // to put the plane farther from center_A.
  const double z_cut_CA_lower =
      z_cutting_plane_ + 1.6 * (center_EA[2] - z_cutting_plane_);
  // z_cut_CA_upper is the upper z_plane position for the CA endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.
  const double z_cut_CA_upper =
      std::max(0.5 * (z_cut_CA_lower + inner_radius_C), 0.7 * inner_radius_C);
  // z_cut_EA_upper is the upper z_plane position for the EA endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.
  const double z_cut_EA_upper = center_A_[2] + 0.7 * outer_radius_A_;
  // z_cut_EA_lower is the lower z_plane position for the EA endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.
  const double z_cut_EA_lower = center_A_[2] - 0.7 * outer_radius_A_;

  // CA Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(aligned),
      CoordinateMaps::UniformCylindricalEndcap(center_EA, make_array<3>(0.0),
                                               radius_EA, inner_radius_C,
                                               z_cut_CA_lower, z_cut_CA_upper),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // CA Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(aligned),
      CoordinateMaps::UniformCylindricalSide(
          // codecov complains about the next line being untested.
          // No idea why, since this entire function is called.
          // LCOV_EXCL_START
          center_EA, make_array<3>(0.0), radius_EA, inner_radius_C,
          // LCOV_EXCL_STOP
          z_cut_CA_lower, z_cutting_plane_, z_cut_CA_upper, z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // EA Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(aligned),
      CoordinateMaps::UniformCylindricalEndcap(center_A_, center_EA,
                                               outer_radius_A_, radius_EA,
                                               z_cut_EA_upper, z_cut_CA_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // EA Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(aligned),
      // For some reason codecov complains about the next line.
      CoordinateMaps::UniformCylindricalSide(  // LCOV_EXCL_LINE
          center_A_, center_EA, outer_radius_A_, radius_EA, z_cut_EA_upper,
          z_cut_EA_lower, z_cut_CA_lower, z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // z_cut_CB_lower is the lower z_plane position for the CB endcap,
  // defined by https://arxiv.org/abs/1206.3015 in the bulleted list
  // after Eq. (A.19) EXCEPT that here we use a factor of 1.6 instead of 1.5
  // to put the plane farther from center_B.
  // Note here that 'lower' means 'farther from z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_CB_lower =
      z_cutting_plane_ + 1.6 * (center_EB[2] - z_cutting_plane_);
  // z_cut_CB_upper is the upper z_plane position for the CB endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme. Note here that 'upper' means 'closer to z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_CB_upper =
      std::min(0.5 * (z_cut_CB_lower - inner_radius_C), -0.7 * inner_radius_C);
  // z_cut_EB_upper is the upper z_plane position for the EB endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.  Note here that 'upper' means 'closer to z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_EB_upper = center_B_[2] - 0.7 * outer_radius_B_;
  // z_cut_EB_lower is the lower z_plane position for the EB endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme. Note here that 'lower' means 'farther from z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_EB_lower = center_B_[2] + 0.7 * outer_radius_B_;

  // EB Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(half_turn_about_zeta),
      CoordinateMaps::UniformCylindricalEndcap(
          flip_about_xy_plane(center_B_), flip_about_xy_plane(center_EB),
          outer_radius_B_, radius_EB, -z_cut_EB_upper, -z_cut_CB_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // EB Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(half_turn_about_zeta),
      CoordinateMaps::UniformCylindricalSide(
          flip_about_xy_plane(center_B_), flip_about_xy_plane(center_EB),
          outer_radius_B_, radius_EB, -z_cut_EB_upper, -z_cut_EB_lower,
          -z_cut_CB_lower, -z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // MA Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(half_turn_about_zeta),
      CoordinateMaps::UniformCylindricalFlatEndcap(
          flip_about_xy_plane(center_A_),
          flip_about_xy_plane(center_cutting_plane), outer_radius_A_, radius_MB,
          -z_cut_EA_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));
  // MB Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(aligned),
      // For some reason codecov complains about the next line.
      CoordinateMaps::UniformCylindricalFlatEndcap(  // LCOV_EXCL_LINE
          center_B_, center_cutting_plane, outer_radius_B_, radius_MB,
          z_cut_EB_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // CB Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(half_turn_about_zeta),
      CoordinateMaps::UniformCylindricalEndcap(
          flip_about_xy_plane(center_EB), make_array<3>(0.0), radius_EB,
          inner_radius_C, -z_cut_CB_lower, -z_cut_CB_upper),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // CB Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(half_turn_about_zeta),
      CoordinateMaps::UniformCylindricalSide(
          flip_about_xy_plane(center_EB), make_array<3>(0.0), radius_EB,
          inner_radius_C, -z_cut_CB_lower, -z_cutting_plane_, -z_cut_CB_upper,
          -z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  const size_t ea_endcap_block = block_positions_.at("EAFilledCylinder");
  const size_t ea_side_block = block_positions_.at("EACylinder");
  const size_t ma_endcap_block = block_positions_.at("MAFilledCylinder");
  const size_t eb_endcap_block = block_positions_.at("EBFilledCylinder");
  const size_t eb_side_block = block_positions_.at("EBCylinder");
  const size_t mb_endcap_block = block_positions_.at("MBFilledCylinder");
  const size_t ca_endcap_block = block_positions_.at("CAFilledCylinder");
  const size_t ca_side_block = block_positions_.at("CACylinder");
  const size_t cb_endcap_block = block_positions_.at("CBFilledCylinder");
  const size_t cb_side_block = block_positions_.at("CBCylinder");

  // Excision spheres
  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};

  std::unordered_map<size_t, Direction<3>> abutting_directions_A;
  if (include_inner_sphere_A_) {
    // LCOV_EXCL_START
    abutting_directions_A.emplace(block_positions_.at("InnerAShell0"),
                                  Direction<3>::lower_xi());
    // LCOV_EXCL_STOP
  } else {
    abutting_directions_A.emplace(ea_endcap_block, Direction<3>::lower_zeta());
    abutting_directions_A.emplace(ma_endcap_block, Direction<3>::lower_zeta());
    abutting_directions_A.emplace(ea_side_block, Direction<3>::lower_xi());
  }
  excision_spheres.emplace(
      "ExcisionSphereA",
      ExcisionSphere<3>{
          radius_A_,
          tnsr::I<double, 3, Frame::Grid>(rotate_from_z_to_x_axis(center_A_)),
          abutting_directions_A});

  std::unordered_map<size_t, Direction<3>> abutting_directions_B;
  if (include_inner_sphere_B_) {
    // LCOV_EXCL_START
    abutting_directions_B.emplace(block_positions_.at("InnerBShell0"),
                                  Direction<3>::lower_xi());
    // LCOV_EXCL_STOP
  } else {
    abutting_directions_B.emplace(eb_endcap_block, Direction<3>::lower_zeta());
    abutting_directions_B.emplace(mb_endcap_block, Direction<3>::lower_zeta());
    abutting_directions_B.emplace(eb_side_block, Direction<3>::lower_xi());
  }
  excision_spheres.emplace(
      "ExcisionSphereB",
      ExcisionSphere<3>{
          radius_B_,
          tnsr::I<double, 3, Frame::Grid>(rotate_from_z_to_x_axis(center_B_)),
          abutting_directions_B});

  Domain<3> domain;
  // non-shell maps
  std::vector<DirectionMap<3, BlockNeighbors<3>>> inner_neighbors{
      coordinate_maps.size()};

  // Add a cylinder as a neighor of a cylinder
  auto add_cyl_cyl_block_neighbor =
      [](std::vector<DirectionMap<3, BlockNeighbors<3>>>& neighbors,
         const size_t this_block_number, const size_t neighbor_block_number,
         const Direction<3>& direction,
         const OrientationMap<3>& orientation_map) {
        neighbors[this_block_number].emplace(
            direction,
            BlockNeighbors<3>{{neighbor_block_number},
                              {{neighbor_block_number, orientation_map}},
                              /*are_conforming=*/true});
      };

  // EA Filled Cylinder
  add_cyl_cyl_block_neighbor(
      inner_neighbors, ea_endcap_block, ea_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::upper_xi()}}});
  add_cyl_cyl_block_neighbor(inner_neighbors, ea_endcap_block, ca_endcap_block,
                             Direction<3>::upper_zeta(), aligned);

  // EA Cylinder
  add_cyl_cyl_block_neighbor(
      inner_neighbors, ea_side_block, ea_endcap_block,
      Direction<3>::upper_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::lower_xi()}}});
  add_cyl_cyl_block_neighbor(
      inner_neighbors, ea_side_block, ma_endcap_block,
      Direction<3>::lower_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::lower_eta(),
                          Direction<3>::upper_xi()}}});
  add_cyl_cyl_block_neighbor(inner_neighbors, ea_side_block, ca_side_block,
                             Direction<3>::upper_xi(), aligned);

  // MA Filled Cylinder
  add_cyl_cyl_block_neighbor(
      inner_neighbors, ma_endcap_block, ea_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::lower_eta(),
                          Direction<3>::upper_xi()}}});
  add_cyl_cyl_block_neighbor(
      inner_neighbors, ma_endcap_block, mb_endcap_block,
      Direction<3>::upper_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                          Direction<3>::lower_zeta()}}});

  // CA Filled Cylinder
  add_cyl_cyl_block_neighbor(
      inner_neighbors, ca_endcap_block, ca_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::upper_xi()}}});
  add_cyl_cyl_block_neighbor(inner_neighbors, ca_endcap_block, ea_endcap_block,
                             Direction<3>::lower_zeta(), aligned);

  // CA Cylinder
  add_cyl_cyl_block_neighbor(
      inner_neighbors, ca_side_block, ca_endcap_block,
      Direction<3>::upper_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::lower_xi()}}});
  add_cyl_cyl_block_neighbor(
      inner_neighbors, ca_side_block, cb_side_block, Direction<3>::lower_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                          Direction<3>::lower_zeta()}}});
  add_cyl_cyl_block_neighbor(inner_neighbors, ca_side_block, ea_side_block,
                             Direction<3>::lower_xi(), aligned);

  // EB Filled Cylinder
  add_cyl_cyl_block_neighbor(
      inner_neighbors, eb_endcap_block, eb_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::upper_xi()}}});
  add_cyl_cyl_block_neighbor(inner_neighbors, eb_endcap_block, cb_endcap_block,
                             Direction<3>::upper_zeta(), aligned);

  // EB Cylinder
  add_cyl_cyl_block_neighbor(
      inner_neighbors, eb_side_block, eb_endcap_block,
      Direction<3>::upper_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::lower_xi()}}});
  add_cyl_cyl_block_neighbor(
      inner_neighbors, eb_side_block, mb_endcap_block,
      Direction<3>::lower_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::lower_eta(),
                          Direction<3>::upper_xi()}}});
  add_cyl_cyl_block_neighbor(inner_neighbors, eb_side_block, cb_side_block,
                             Direction<3>::upper_xi(), aligned);

  // MB Filled Cylinder
  add_cyl_cyl_block_neighbor(
      inner_neighbors, mb_endcap_block, eb_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::lower_eta(),
                          Direction<3>::upper_xi()}}});
  add_cyl_cyl_block_neighbor(
      inner_neighbors, mb_endcap_block, ma_endcap_block,
      Direction<3>::upper_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                          Direction<3>::lower_zeta()}}});

  // CB Filled Cylinder
  add_cyl_cyl_block_neighbor(
      inner_neighbors, cb_endcap_block, cb_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::upper_xi()}}});
  add_cyl_cyl_block_neighbor(inner_neighbors, cb_endcap_block, eb_endcap_block,
                             Direction<3>::lower_zeta(), aligned);

  // CB Cylinder
  add_cyl_cyl_block_neighbor(
      inner_neighbors, cb_side_block, cb_endcap_block,
      Direction<3>::upper_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::lower_xi()}}});
  add_cyl_cyl_block_neighbor(
      inner_neighbors, cb_side_block, ca_side_block, Direction<3>::lower_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                          Direction<3>::lower_zeta()}}});
  add_cyl_cyl_block_neighbor(inner_neighbors, cb_side_block, eb_side_block,
                             Direction<3>::lower_xi(), aligned);

  // Connect the E sphere cylinder blocks to the outermost inner shells and
  // connect the C sphere cylinder blocks to the innermost outer shell
  const OrientationMap<3> shell_to_cyl_endcap{
      {{Direction<3>::upper_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  const auto cyl_endcap_to_shell =
      shell_to_cyl_endcap.inverse_map();
  const OrientationMap<3> shell_to_cyl_side{
      {{Direction<3>::upper_xi(), Direction<3>::self(), Direction<3>::self()}}};
  const auto cyl_side_to_shell = shell_to_cyl_side.inverse_map();

  // Add a spherical shell as a neighor of a cylinder
  auto add_cyl_shell_block_neighbor =
      [](std::vector<DirectionMap<3, BlockNeighbors<3>>>& neighbors,
         const bool cyl_is_filled, const bool shell_is_outside_cyl,
         const size_t cyl_block_number, const size_t shell_block_number,
         const OrientationMap<3>& orientation_map) {
        const Direction<3> direction =
            cyl_is_filled ? (shell_is_outside_cyl ? Direction<3>::upper_zeta()
                                                  : Direction<3>::lower_zeta())
                          : (shell_is_outside_cyl ? Direction<3>::upper_xi()
                                                  : Direction<3>::lower_xi());

        neighbors[cyl_block_number].emplace(
            direction,
            BlockNeighbors<3>{{shell_block_number},
                              {{shell_block_number, orientation_map}},
                              /*are_conforming=*/false});
      };

  // Get the number of spherical shells for InnerSphereA and InnerSphereB
  const size_t num_shells_inner_sphere_A =
      not include_inner_sphere_A_ ? 0 : block_groups_.at("InnerSphereA").size();
  if (include_inner_sphere_A_) {
    ASSERT(num_shells_inner_sphere_A > 0,
           "Requested to include InnerSphereA but the number of spherical "
           "shells for InnerSphereA is 0.");
  } else {
    ASSERT(not block_groups_.contains("InnerSphereA"),
           "Did not request to include InnerSphereA, but InnerSphereA found as "
           "an existing block group.");
  }
  const size_t num_shells_inner_sphere_B =
      not include_inner_sphere_B_ ? 0 : block_groups_.at("InnerSphereB").size();
  if (include_inner_sphere_B_) {
    ASSERT(num_shells_inner_sphere_B > 0,
           "Requested to include InnerSphereB but the number of spherical "
           "shells for InnerSphereB is 0.");
  } else {
    ASSERT(not block_groups_.contains("InnerSphereB"),
           "Did not request to include InnerSphereB, but InnerSphereB found as "
           "an existing block group.");
  }

  if (include_inner_sphere_A_) {
    // // Connect InnerSphereA shells to each other
    // for (size_t shell_number = 1; shell_number < num_shells_inner_sphere_B;
    //      shell_number++) {
    //   const size_t inner_shell_block_number = block_positions_.at(
    //       std::string("InnerAShell").append(std::to_string(shell_number -
    //       1)));
    //   const size_t outer_shell_block_number = block_positions_.at(
    //       std::string("InnerAShell").append(std::to_string(shell_number -
    //       1)));

    //   add_spherical_shell_block_neighbors(
    //       inner_neighbors, inner_shell_block_number,
    //       outer_shell_block_number);
    // }

    // // Connect outermost shell of InnerSphereA shells to cylinders

    // Get outermost block of InnerSphereA
    const std::string outermost_inner_shell_A_block_name =
        std::string("InnerAShell")
            .append(std::to_string(num_shells_inner_sphere_A - 1));
    const size_t outermost_inner_shell_A_block_number =
        block_positions_.at(outermost_inner_shell_A_block_name);

    // EA Filled Cylinder
    add_cyl_shell_block_neighbor(inner_neighbors, true, false, ea_endcap_block,
                                 outermost_inner_shell_A_block_number,
                                 cyl_endcap_to_shell);
    // EA Cylinder
    add_cyl_shell_block_neighbor(inner_neighbors, false, false, ea_side_block,
                                 outermost_inner_shell_A_block_number,
                                 cyl_side_to_shell);
    // MA Filled Cylinder
    add_cyl_shell_block_neighbor(inner_neighbors, true, false, ma_endcap_block,
                                 outermost_inner_shell_A_block_number,
                                 cyl_endcap_to_shell);
  }

  if (include_inner_sphere_B_) {
    // // Connect InnerSphereB shells to each other
    // for (size_t shell_number = 1; shell_number < num_shells_inner_sphere_B;
    //      shell_number++) {
    //   const size_t inner_shell_block_number = block_positions_.at(
    //       std::string("InnerBShell").append(std::to_string(shell_number -
    //       1)));
    //   const size_t outer_shell_block_number = block_positions_.at(
    //       std::string("InnerBShell").append(std::to_string(shell_number -
    //       1)));

    //   add_spherical_shell_block_neighbors(
    //       inner_neighbors, inner_shell_block_number,
    //       outer_shell_block_number);
    // }

    // // Connect outermost shell of InnerSphereB shells to cylinders

    // Get outermost block of InnerSphereB
    const std::string outermost_inner_shell_B_block_name =
        std::string("InnerBShell")
            .append(std::to_string(num_shells_inner_sphere_B - 1));
    const size_t outermost_inner_shell_B_block_number =
        block_positions_.at(outermost_inner_shell_B_block_name);

    // EB Filled Cylinder
    add_cyl_shell_block_neighbor(inner_neighbors, true, false, eb_endcap_block,
                                 outermost_inner_shell_B_block_number,
                                 cyl_endcap_to_shell);
    // EB Cylinder
    add_cyl_shell_block_neighbor(inner_neighbors, false, false, eb_side_block,
                                 outermost_inner_shell_B_block_number,
                                 cyl_side_to_shell);
    // MB Filled Cylinder
    add_cyl_shell_block_neighbor(inner_neighbors, true, false, mb_endcap_block,
                                 outermost_inner_shell_B_block_number,
                                 cyl_endcap_to_shell);
  }

  const size_t outer_shell_block = block_positions_.at("OuterShell0");

  // CA Filled Cylinder
  add_cyl_shell_block_neighbor(inner_neighbors, true, true, ca_endcap_block,
                               outer_shell_block, cyl_endcap_to_shell);
  // CA Cylinder
  add_cyl_shell_block_neighbor(inner_neighbors, false, true, ca_side_block,
                               outer_shell_block, cyl_side_to_shell);

  // CB Filled Cylinder
  add_cyl_shell_block_neighbor(inner_neighbors, true, true, cb_endcap_block,
                               outer_shell_block, cyl_endcap_to_shell);
  // CB Cylinder
  add_cyl_shell_block_neighbor(inner_neighbors, false, true, cb_side_block,
                               outer_shell_block, cyl_side_to_shell);

  // Build blocks in final order.
  std::vector<Block<3>> blocks;
  blocks.reserve(number_of_blocks_);

  // (a) Inner cylindrical blocks.
  for (const auto& [name, position] : block_positions_) {
    if (name.find("Cylinder") != std::string::npos) {
      const auto cyl_topology = name.find("Filled") != std::string::npos
                                    ? domain::topologies::full_cylinder
                                    : domain::topologies::cylindrical_shell;
      blocks.emplace_back(std::move(coordinate_maps[position]), position,
                          std::move(inner_neighbors[position]), name,
                          cyl_topology);
    }
  }

  // Add a cylindrical endcap as a neighbor of a spherical shell
  auto add_shell_cyl_endcap_neighbor =
      [&shell_to_cyl_endcap](
          std::unordered_set<size_t>& cyl_ids,
          std::unordered_map<size_t, OrientationMap<3>>& cyl_orientations,
          const size_t cyl_block_number) {
        cyl_ids.insert(cyl_block_number);
        cyl_orientations.emplace(cyl_block_number, shell_to_cyl_endcap);
      };

  // Add a cylindrical side as a neighbor of a spherical shell
  auto add_shell_cyl_side_neighbor =
      [&shell_to_cyl_side](
          std::unordered_set<size_t>& cyl_ids,
          std::unordered_map<size_t, OrientationMap<3>>& cyl_orientations,
          const size_t cyl_block_number) {
        cyl_ids.insert(cyl_block_number);
        cyl_orientations.emplace(cyl_block_number, shell_to_cyl_side);
      };

  // Add nested spherical shells as block neighbors
  auto add_spherical_shell_block_neighbors =
      [&aligned](std::vector<DirectionMap<3, BlockNeighbors<3>>>& neighbors,
                 const size_t inner_shell_block_number,
                 const size_t outer_shell_block_number) {
        neighbors[inner_shell_block_number].emplace(
            Direction<3>::upper_xi(),
            BlockNeighbors<3>{{outer_shell_block_number},
                              {{outer_shell_block_number, aligned}},
                              /*are_conforming=*/true});
        neighbors[outer_shell_block_number].emplace(
            Direction<3>::lower_xi(),
            BlockNeighbors<3>{{inner_shell_block_number},
                              {{inner_shell_block_number, aligned}},
                              /*are_conforming=*/true});
      };

  using Affine = CoordinateMaps::Affine;
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

  // TODO : make new code adding and connecting multiple shells better

  // (b) SH inner shell blocks for InnerSphereA.
  if (include_inner_sphere_A_) {
    // Connect InnerSphereA shells to each other
    const double distance_to_cutting_plane =
        fabs(z_cutting_plane_ - gsl::at(center_A_, 2));
    const double distance_to_cutting_plane_over_inner_radius =
        distance_to_cutting_plane / radius_A_;
    const double coef =
        pow(distance_to_cutting_plane_over_inner_radius,
            1.0 / static_cast<double>(num_shells_inner_sphere_A + 1));
    double inner_radius = radius_A_;
    double outer_radius = inner_radius;
    std::vector<DirectionMap<3, BlockNeighbors<3>>> inner_a_sh_neighbors{2_st};

    for (size_t shell_number = 0; shell_number + 1 < num_shells_inner_sphere_A;
         shell_number++) {
      const size_t inner_shell_block_number = block_positions_.at(
          std::string("InnerAShell").append(std::to_string(shell_number)));
      const size_t outer_shell_block_number = block_positions_.at(
          std::string("InnerAShell").append(std::to_string(shell_number + 1)));

      add_spherical_shell_block_neighbors(inner_a_sh_neighbors,
                                          inner_shell_block_number,
                                          outer_shell_block_number);

      outer_radius *= coef;
      auto inner_a_sh_map = make_spherical_shell_coord_map(
          inner_radius, outer_radius, rotate_from_z_to_x_axis(center_A_));

      blocks.emplace_back(
          std::move(inner_a_sh_map), outer_shell_block_number,
          std::move(inner_a_sh_neighbors[outer_shell_block_number]),
          gsl::at(block_names_, outer_shell_block_number),
          domain::topologies::spherical_shell);
      inner_radius = outer_radius;
    }

    // Connect outermost shell of InnerSphereA shells to cylinders

    // upper_xi → EA endcap, EA side, and MA blocks
    // (non-conforming, multi-neighbor).
    std::unordered_set<size_t> inner_a_cyl_ids;
    std::unordered_map<size_t, OrientationMap<3>> inner_a_cyl_orientations;

    // EA Filled Cylinder
    add_shell_cyl_endcap_neighbor(inner_a_cyl_ids, inner_a_cyl_orientations,
                                  ea_endcap_block);
    // EA Cylinder
    add_shell_cyl_side_neighbor(inner_a_cyl_ids, inner_a_cyl_orientations,
                                ea_side_block);
    // MA Filled Cylinder
    add_shell_cyl_endcap_neighbor(inner_a_cyl_ids, inner_a_cyl_orientations,
                                  ma_endcap_block);

    auto outermost_inner_a_sh_map = make_spherical_shell_coord_map(
        inner_radius, outer_radius_A_, rotate_from_z_to_x_axis(center_A_));

    // Get outermost block of InnerSphereA
    const std::string outermost_inner_shell_A_block_name =
        std::string("InnerAShell")
            .append(std::to_string(num_shells_inner_sphere_A - 1));
    const size_t outermost_inner_shell_A_block_number =
        block_positions_.at(outermost_inner_shell_A_block_name);

    DirectionMap<3, BlockNeighbors<3>> outermost_inner_a_sh_neighbors;
    outermost_inner_a_sh_neighbors.emplace(
        Direction<3>::upper_xi(),
        BlockNeighbors<3>{std::move(inner_a_cyl_ids),
                          std::move(inner_a_cyl_orientations),
                          /*are_conforming=*/false});
    blocks.emplace_back(std::move(outermost_inner_a_sh_map),
                        outermost_inner_shell_A_block_number,
                        std::move(outermost_inner_a_sh_neighbors),
                        block_names_[outermost_inner_shell_A_block_number],
                        domain::topologies::spherical_shell);
  }

  // (c) SH inner shell blocks for InnerSphereB.
  if (include_inner_sphere_B_) {
    // Connect InnerSphereB shells to each other
    const double distance_to_cutting_plane =
        fabs(z_cutting_plane_ - gsl::at(center_B_, 2));
    const double distance_to_cutting_plane_over_inner_radius =
        distance_to_cutting_plane / radius_B_;
    const double coef =
        pow(distance_to_cutting_plane_over_inner_radius,
            1.0 / static_cast<double>(num_shells_inner_sphere_B + 1));
    double inner_radius = radius_B_;
    double outer_radius = inner_radius;
    std::vector<DirectionMap<3, BlockNeighbors<3>>> inner_b_sh_neighbors{2_st};

    for (size_t shell_number = 0; shell_number + 1 < num_shells_inner_sphere_B;
         shell_number++) {
      const size_t inner_shell_block_number = block_positions_.at(
          std::string("InnerBShell").append(std::to_string(shell_number)));
      const size_t outer_shell_block_number = block_positions_.at(
          std::string("InnerBShell").append(std::to_string(shell_number + 1)));

      add_spherical_shell_block_neighbors(inner_b_sh_neighbors,
                                          inner_shell_block_number,
                                          outer_shell_block_number);

      outer_radius *= coef;
      auto inner_b_sh_map = make_spherical_shell_coord_map(
          inner_radius, outer_radius, rotate_from_z_to_x_axis(center_B_));

      blocks.emplace_back(
          std::move(inner_b_sh_map), outer_shell_block_number,
          std::move(inner_b_sh_neighbors[outer_shell_block_number]),
          gsl::at(block_names_, outer_shell_block_number),
          domain::topologies::spherical_shell);
      inner_radius = outer_radius;
    }

    // Connect outermost shell of InnerSphereB shells to cylinders

    // upper_xi → EB endcap, EB side, and MB blocks
    // (non-conforming, multi-neighbor).
    std::unordered_set<size_t> inner_b_cyl_ids;
    std::unordered_map<size_t, OrientationMap<3>> inner_b_cyl_orientations;

    // EB Filled Cylinder
    add_shell_cyl_endcap_neighbor(inner_b_cyl_ids, inner_b_cyl_orientations,
                                  eb_endcap_block);
    // EB Cylinder
    add_shell_cyl_side_neighbor(inner_b_cyl_ids, inner_b_cyl_orientations,
                                eb_side_block);
    // MB Filled Cylinder
    add_shell_cyl_endcap_neighbor(inner_b_cyl_ids, inner_b_cyl_orientations,
                                  mb_endcap_block);

    auto outermost_inner_b_sh_map = make_spherical_shell_coord_map(
        inner_radius, outer_radius_B_, rotate_from_z_to_x_axis(center_B_));

    // Get outermost block of InnerSphereA
    const std::string outermost_inner_shell_B_block_name =
        std::string("InnerBShell")
            .append(std::to_string(num_shells_inner_sphere_B - 1));
    const size_t outermost_inner_shell_B_block_number =
        block_positions_.at(outermost_inner_shell_B_block_name);

    DirectionMap<3, BlockNeighbors<3>> outermost_inner_b_sh_neighbors;
    outermost_inner_b_sh_neighbors.emplace(
        Direction<3>::upper_xi(),
        BlockNeighbors<3>{std::move(inner_b_cyl_ids),
                          std::move(inner_b_cyl_orientations),
                          /*are_conforming=*/false});
    blocks.emplace_back(std::move(outermost_inner_b_sh_map),
                        outermost_inner_shell_B_block_number,
                        std::move(outermost_inner_b_sh_neighbors),
                        block_names_[outermost_inner_shell_B_block_number],
                        domain::topologies::spherical_shell);
  }

  // (d) SH outer shell blocks for OuterSphere.
  // lower_xi → all 4 CA, CB blocks (non-conforming, multi-neighbor).
  std::unordered_set<size_t> outer_cyl_ids;
  std::unordered_map<size_t, OrientationMap<3>> outer_cyl_orientations;

  // CA Filled Cylinder
  add_shell_cyl_endcap_neighbor(outer_cyl_ids, outer_cyl_orientations,
                                ca_endcap_block);
  // CA Cylinder
  add_shell_cyl_side_neighbor(outer_cyl_ids, outer_cyl_orientations,
                              ca_side_block);
  // CB Filled Cylinder
  add_shell_cyl_endcap_neighbor(outer_cyl_ids, outer_cyl_orientations,
                                cb_endcap_block);
  // CB Cylinder
  add_shell_cyl_side_neighbor(outer_cyl_ids, outer_cyl_orientations,
                              cb_side_block);

  auto outer_sh_map = make_spherical_shell_coord_map(
      inner_radius_C, outer_radius_, make_array<3>(0.0));

  DirectionMap<3, BlockNeighbors<3>> outer_sh_neighbors;
  outer_sh_neighbors.emplace(
      Direction<3>::lower_xi(),
      BlockNeighbors<3>{std::move(outer_cyl_ids),
                        std::move(outer_cyl_orientations),
                        /*are_conforming=*/false});

  blocks.emplace_back(
      std::move(outer_sh_map), outer_shell_block, std::move(outer_sh_neighbors),
      block_names_[outer_shell_block], domain::topologies::spherical_shell);

  domain =
      Domain<3>{std::move(blocks), std::move(excision_spheres), block_groups_};

  if (time_dependent_options_.has_value()) {
    ASSERT(include_inner_sphere_A_ and include_inner_sphere_B_,
           "When using time dependent maps for the CylindricalBBH domain, you "
           "must include both inner spheres.");
    const size_t first_inner_shell_A_block =
        block_positions_.at("InnerAShell0");
    const size_t first_inner_shell_B_block =
        block_positions_.at("InnerBShell0");

    // Default initialize everything to nullptr so that we only need to set the
    // appropriate block maps for the specific frames
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        grid_to_inertial_block_maps{number_of_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
        grid_to_distorted_block_maps{number_of_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
        distorted_to_inertial_block_maps{number_of_blocks_};

    // The 0th block always exists and will only need an rigid expansion +
    // rotation + translation map from the grid to inertial frame. No maps to
    // the distorted frame
    grid_to_inertial_block_maps[0] =
        time_dependent_options_
            ->grid_to_inertial_map<domain::ObjectLabel::None>(false, true);

    // The first block in the outer shell needs the transition expansion +
    // rotation + translation map from the grid to inertial frame. No maps to
    // the distorted frame
    grid_to_inertial_block_maps[outer_shell_block] =
        time_dependent_options_
            ->grid_to_inertial_map<domain::ObjectLabel::None>(false, false);

    // Inside the excision sphere we add the grid to inertial map from the
    // outer shell. This allows the center of the excisions/horizons to be
    // mapped properly to the inertial frame.
    domain.inject_time_dependent_map_for_excision_sphere(
        "ExcisionSphereA",
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::A>(
            true, true, true));
    domain.inject_time_dependent_map_for_excision_sphere(
        "ExcisionSphereB",
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::B>(
            true, true, true));

    // The `true` being passed to the functions specifies that the size map
    // *should* be included in the distorted frame.
    grid_to_inertial_block_maps[first_inner_shell_A_block] =
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::A>(
            true, true);
    grid_to_distorted_block_maps[first_inner_shell_A_block] =
        time_dependent_options_->grid_to_distorted_map<domain::ObjectLabel::A>(
            true);
    distorted_to_inertial_block_maps[first_inner_shell_A_block] =
        time_dependent_options_
            ->distorted_to_inertial_map<domain::ObjectLabel::A>(true, true);

    grid_to_inertial_block_maps[first_inner_shell_B_block] =
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::B>(
            true, true);
    grid_to_distorted_block_maps[first_inner_shell_B_block] =
        time_dependent_options_->grid_to_distorted_map<domain::ObjectLabel::B>(
            true);
    distorted_to_inertial_block_maps[first_inner_shell_B_block] =
        time_dependent_options_
            ->distorted_to_inertial_map<domain::ObjectLabel::B>(true, true);

    for (size_t block = 1; block < number_of_blocks_; ++block) {
      if (block == first_inner_shell_A_block or
          block == first_inner_shell_B_block or block == outer_shell_block) {
        continue;  // Already initialized
      } else if (block > first_inner_shell_A_block and
                 block < first_inner_shell_B_block) {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[first_inner_shell_A_block]->get_clone();
        if (grid_to_distorted_block_maps[first_inner_shell_A_block] !=
            nullptr) {
          grid_to_distorted_block_maps[block] =
              grid_to_distorted_block_maps[first_inner_shell_A_block]
                  ->get_clone();
          distorted_to_inertial_block_maps[block] =
              distorted_to_inertial_block_maps[first_inner_shell_A_block]
                  ->get_clone();
        }
      } else if (block > first_inner_shell_B_block and
                 block < outer_shell_block) {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[first_inner_shell_B_block]->get_clone();
        if (grid_to_distorted_block_maps[first_inner_shell_B_block] !=
            nullptr) {
          grid_to_distorted_block_maps[block] =
              grid_to_distorted_block_maps[first_inner_shell_B_block]
                  ->get_clone();
          distorted_to_inertial_block_maps[block] =
              distorted_to_inertial_block_maps[first_inner_shell_B_block]
                  ->get_clone();
        }
      } else if (block > outer_shell_block) {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[outer_shell_block]->get_clone();
      } else {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[0]->get_clone();
      }
    }

    for (size_t block = 0; block < number_of_blocks_; ++block) {
      domain.inject_time_dependent_map_for_block(
          block, std::move(grid_to_inertial_block_maps[block]),
          std::move(grid_to_distorted_block_maps[block]),
          std::move(distorted_to_inertial_block_maps[block]));
    }
  }

  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
CylindricalBinaryCompactObject::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{number_of_blocks_};
  const size_t ea_endcap_block = block_positions_.at("EAFilledCylinder");
  const size_t ea_side_block = block_positions_.at("EACylinder");
  const size_t ma_endcap_block = block_positions_.at("MAFilledCylinder");
  const size_t eb_endcap_block = block_positions_.at("EBFilledCylinder");
  const size_t eb_side_block = block_positions_.at("EBCylinder");
  const size_t mb_endcap_block = block_positions_.at("MBFilledCylinder");
  const size_t outer_shell_block = block_positions_.at("OuterShell0");

  if (not include_inner_sphere_A_) {
      // EA Filled Cylinder
      boundary_conditions[ea_endcap_block][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
      // MA Filled Cylinder
      boundary_conditions[ma_endcap_block][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
      // EA Cylinder
      boundary_conditions[ea_side_block][Direction<3>::lower_xi()] =
          inner_boundary_condition_->get_clone();
  } else {
    boundary_conditions[block_positions_.at("InnerAShell0")]
                       [Direction<3>::lower_xi()] =
                           inner_boundary_condition_->get_clone();
  }
  if (not include_inner_sphere_B_) {
      // EB Filled Cylinder
      boundary_conditions[eb_endcap_block][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
      // MB Filled Cylinder
      boundary_conditions[mb_endcap_block][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
      // EB Cylinder
      boundary_conditions[eb_side_block][Direction<3>::lower_xi()] =
          inner_boundary_condition_->get_clone();
  } else {
    boundary_conditions[block_positions_.at("InnerBShell0")]
                       [Direction<3>::lower_xi()] =
                           inner_boundary_condition_->get_clone();
  }
  boundary_conditions[outer_shell_block][Direction<3>::upper_xi()] =
      outer_boundary_condition_->get_clone();

  return boundary_conditions;
}

std::vector<std::array<size_t, 3>>
CylindricalBinaryCompactObject::initial_extents() const {
  return initial_grid_points_;
}

std::vector<std::array<size_t, 3>>
CylindricalBinaryCompactObject::initial_refinement_levels() const {
  return initial_refinement_;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
CylindricalBinaryCompactObject::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  return time_dependent_options_.has_value()
             ? time_dependent_options_->create_functions_of_time(
                   initial_expiration_times)
             : std::unordered_map<
                   std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{};
}
}  // namespace domain::creators
