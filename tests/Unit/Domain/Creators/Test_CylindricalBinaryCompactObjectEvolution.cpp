// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Structure/ZCurve.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t Dim = 3;

using CylBCO = ::domain::creators::CylindricalBinaryCompactObject;
using TimeDepOptions = domain::creators::bco::TimeDependentMapOptions<true>;
using RefinementMap = std::unordered_map<std::string, size_t>;
using GridPointsMap = std::unordered_map<
    std::string, std::variant<std::array<size_t, Dim>, std::array<size_t, 2>>>;

// using params from run at:
//     /home/almacedo/runs/bbh/test/cbco_use_cylinders_fix_angular_orientation/b2678025e34bc6bdf2454ff623a6a921bd0a91b3/evolve/gh/01/Segment_0000/Inspiral.yaml
TimeDepOptions construct_time_dependent_options() {
  constexpr double initial_time = 0.0;

  // ExpansionMap
  const std::array<double, Dim> initial_expansion_values{
      1.0, -5.307865406986267e-05, 0.0};
  const double decay_timescale_outer_boundary = 50.0;
  const double asymptotic_velocity_outer_boundary = -1.0e-6;

  // RotationMap
  const std::array<double, Dim> initial_angular_velocity{0.0, 0.0,
                                                         0.015873788863023465};

  // TranslationMap
  const std::array<std::array<double, Dim>, Dim> initial_translation_values{
      std::array{0.0, 0.0, 0.0}, std::array{0.0, 0.0, 0.0},
      std::array{0.0, 0.0, 0.0}};

  // ShapeMap
  const size_t l_max = 10;
  using InitialShapeValues = std::optional<std::variant<
      domain::creators::time_dependent_options::KerrSchildFromBoyerLindquist,
      domain::creators::time_dependent_options::YlmsFromFile,
      domain::creators::time_dependent_options::YlmsFromSpEC>>;
  const InitialShapeValues initial_shape_values = std::nullopt;  // Spherical
  const std::array<double, Dim> initial_size_A_coefs{0.0, 0.0, 0.0};
  const std::array<double, Dim> initial_size_B_coefs{0.0, 0.0, 0.0};
  const double coefficient_truncation_limit = 0.0;
  constexpr bool transition_ends_at_cube = false;

  return TimeDepOptions{
      initial_time,
      domain::creators::time_dependent_options::ExpansionMapOptions<false>{
          initial_expansion_values, decay_timescale_outer_boundary,
          asymptotic_velocity_outer_boundary},
      domain::creators::time_dependent_options::RotationMapOptions<false>{
          initial_angular_velocity},
      domain::creators::time_dependent_options::TranslationMapOptions<Dim>{
          initial_translation_values},
      /*SkewMap=*/std::nullopt,
      domain::creators::time_dependent_options::ShapeMapOptions<
          transition_ends_at_cube, domain::ObjectLabel::A>{
          l_max, initial_shape_values, initial_size_A_coefs,
          coefficient_truncation_limit},
      domain::creators::time_dependent_options::ShapeMapOptions<
          transition_ends_at_cube, domain::ObjectLabel::B>{
          l_max, initial_shape_values, initial_size_B_coefs,
          coefficient_truncation_limit},
      /*GridCenters=*/std::nullopt};
}

// Ok it might be easier to just litter the scalar wave evolution code with
// print statements rather than trying semi-replicate what it does...
void test(const std::array<double, Dim>& center_a,
          const std::array<double, Dim>& center_b, const double radius_a,
          const double radius_b, const bool include_sphere_a,
          const bool include_sphere_b, const double outer_radius,
          const RefinementMap& initial_refinement,
          const GridPointsMap& initial_grid_points,
          const TimeDepOptions& time_dep_options) {
  // Construct boundary conditiions
  auto inner_boundary_condition =
      std::make_unique<gh::BoundaryConditions::DemandOutgoingCharSpeeds<Dim>>();
  auto outer_boundary_condition = std::make_unique<
      gh::BoundaryConditions::ConstraintPreservingBjorhus<Dim>>(
      gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType::
          ConstraintPreservingPhysical,
      std::nullopt);

  // Construct the domain
  const auto creator = ::domain::creators::CylindricalBinaryCompactObject(
      center_a, center_b, radius_a, radius_b, include_sphere_a,
      include_sphere_b, outer_radius, initial_refinement, initial_grid_points,
      time_dep_options, std::move(inner_boundary_condition),
      std::move(outer_boundary_condition));
  const Domain<Dim> domain = creator.create_domain();
  const auto& blocks = domain.blocks();
  const size_t num_blocks = blocks.size();
  const auto& initial_refinement_levels = creator.initial_refinement_levels();
  const auto& initial_extents = creator.initial_extents();
  ASSERT(initial_refinement_levels.size() == num_blocks,
         "initial_refinement_levels.size() != num_blocks");
  ASSERT(initial_extents.size() == num_blocks,
         "initial_extents.size() != num_blocks");

  // Get the total number of elements and by block
  //
  // Taken from BlockZCurveProcDistribution in
  // src/Domain/ElementDistribution.cpp
  size_t num_elements = 0;
  std::vector<size_t> num_elements_by_block(num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    const size_t num_elements_current_block = two_to_the(alg::accumulate(
        initial_refinement_levels[i], 0_st, std::plus<size_t>()));
    num_elements_by_block[i] = num_elements_current_block;
    num_elements += num_elements_current_block;
  }

  // Get the element ids by block
  //
  // Taken from BlockZCurveProcDistribution in
  // src/Domain/ElementDistribution.cpp
  std::vector<std::vector<ElementId<Dim>>> initial_element_ids_by_block(
      num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    initial_element_ids_by_block[i].reserve(num_elements_by_block[i]);
    initial_element_ids_by_block[i] =
        initial_element_ids(blocks[i].id(), initial_refinement_levels[i]);
    alg::sort(initial_element_ids_by_block[i],
              [](const ElementId<Dim>& lhs, const ElementId<Dim>& rhs) {
                return domain::z_curve_index(lhs) < domain::z_curve_index(rhs);
              });
  }

  const Spectral::Basis i1_basis{Spectral::Basis::Legendre};
  const Spectral::Quadrature i1_quadrature = Spectral::Quadrature::GaussLobatto;

  std::vector<Element<Dim>> elements(num_elements);
  std::vector<Mesh<Dim>> meshes(num_elements);
  std::vector<ElementMap<Dim, Frame::Grid>> element_maps(num_elements);
  std::vector<std::unique_ptr<
      ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>>>
      grid_to_inertial_maps(num_elements);
  std::vector<DirectionalIdMap<Dim, Mesh<Dim>>> neighbor_meshes(num_elements);

  // Now, create initial elements and their meshes like in
  // src/Evolution/Initialization/DgDomain.hpp
  //
  // Note: would honestly be ideal to just call
  size_t flattened_element_index = 0;
  for (size_t block_number = 0; block_number < num_blocks; block_number++) {
    const size_t num_elements_this_block =
        initial_element_ids_by_block[block_number].size();
    for (size_t i = 0; i < num_elements_this_block; i++) {
      const ElementId<Dim> element_id =
          initial_element_ids_by_block[block_number][i];

      elements[flattened_element_index] = ::domain::create_initial_element(
          element_id, domain.blocks(), initial_refinement_levels);
      meshes[flattened_element_index] = ::domain::create_initial_mesh(
          initial_extents, elements[flattened_element_index], i1_basis,
          i1_quadrature);
      const auto& my_block = domain.blocks()[element_id.block_id()];
      element_maps[flattened_element_index] =
          ElementMap<Dim, Frame::Grid>{element_id, my_block};

      if (my_block.is_time_dependent()) {
        grid_to_inertial_maps[flattened_element_index] =
            my_block.moving_mesh_grid_to_inertial_map().get_clone();
      } else {
        grid_to_inertial_maps[flattened_element_index] =
            ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
                ::domain::CoordinateMaps::Identity<Dim>{});
      }

      flattened_element_index++;
    }
  }

  ASSERT(flattened_element_index == num_elements,
         "flattened_element_index != num_elements");

  // Now, compare logical and inertial points on boundaries of interest,
  // taking care to apply orientation maps where needed

  CHECK(true);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.CylindricalBinaryCompactObjectEvolution",
    "[Domain][Unit]") {
  // using most of the params from run at:
  //     /home/almacedo/runs/bbh/test/cbco_use_cylinders_fix_angular_orientation/b2678025e34bc6bdf2454ff623a6a921bd0a91b3/evolve/gh/01/Segment_0000/Inspiral.yaml
  //
  // However, using different initial grid points and refinement to inspect
  // grid point matching at boundaries and communication between them.
  // Removing h refinement eliminates complexity and a potential bug source and
  // makes it possible to line up grid points at shared boundaries by setting
  // the grid points to be the same. Having grid points on shared faces that
  // line up will make debugging easier, hopefully.
  const std::array<double, Dim> center_a{7.5, 0.0, 0.0};
  const std::array<double, Dim> center_b{-7.5, 0.0, 0.0};
  const double radius_a = 0.7885647805630362;
  const double radius_b = radius_a;
  const bool include_sphere_a = true;
  const bool include_sphere_b = true;
  const double outer_radius = 600.0;

  const size_t cylinder_refinement = 0;
  const size_t sphere_refinement = 0;

  const std::array<size_t, 2> filled_cylinder_extents{5, 5};
  const std::array<size_t, 3> hollow_cylinder_extents{
      filled_cylinder_extents[2], 4 * filled_cylinder_extents[0] - 3,
      filled_cylinder_extents[0]};
  const std::array<size_t, 2> inner_sphere_extents{12, 20};
  const std::array<size_t, 2> outer_sphere_extents{20, 13};

  const RefinementMap initial_refinement{
      {"CAFilledCylinder", cylinder_refinement},
      {"CBFilledCylinder", cylinder_refinement},
      {"EAFilledCylinder", cylinder_refinement},
      {"EBFilledCylinder", cylinder_refinement},
      {"MAFilledCylinder", cylinder_refinement},
      {"MBFilledCylinder", cylinder_refinement},
      {"CACylinder", cylinder_refinement},
      {"CBCylinder", cylinder_refinement},
      {"EACylinder", cylinder_refinement},
      {"EBCylinder", cylinder_refinement},
      {"InnerSphereA", sphere_refinement},
      {"InnerSphereB", sphere_refinement},
      {"OuterSphere", sphere_refinement}};

  const GridPointsMap initial_grid_points{
      {"CAFilledCylinder", filled_cylinder_extents},
      {"CBFilledCylinder", filled_cylinder_extents},
      {"EAFilledCylinder", filled_cylinder_extents},
      {"EBFilledCylinder", filled_cylinder_extents},
      {"MAFilledCylinder", filled_cylinder_extents},
      {"MBFilledCylinder", filled_cylinder_extents},
      {"CACylinder", hollow_cylinder_extents},
      {"CBCylinder", hollow_cylinder_extents},
      {"EACylinder", hollow_cylinder_extents},
      {"EBCylinder", hollow_cylinder_extents},
      {"InnerSphereA", inner_sphere_extents},
      {"InnerSphereB", inner_sphere_extents},
      {"OuterSphere", outer_sphere_extents}};

  const TimeDepOptions time_dep_options = construct_time_dependent_options();

  test(center_a, center_b, radius_a, radius_b, include_sphere_a,
       include_sphere_b, outer_radius, initial_refinement, initial_grid_points,
       time_dep_options);
}
