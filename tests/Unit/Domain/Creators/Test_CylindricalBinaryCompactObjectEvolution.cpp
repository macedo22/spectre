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

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
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
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Structure/ZCurve.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
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

struct InterfaceTestField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

tnsr::I<DataVector, Dim, Frame::ElementLogical>
element_logical_coordinates_on_mortar(
    const Mesh<Dim - 1>& mortar_mesh, const Direction<Dim>& direction,
    const std::array<Spectral::SegmentSize, Dim - 1>& mortar_size) {
  auto result = interface_logical_coordinates(mortar_mesh, direction);
  size_t mortar_dimension = 0;
  for (size_t d = 0; d < Dim; ++d) {
    if (d == direction.dimension()) {
      continue;
    }
    const auto segment_size = mortar_size[mortar_dimension];
    if (segment_size == Spectral::SegmentSize::LowerHalf) {
      result.get(d) = 0.5 * (result.get(d) - 1.0);
    } else if (segment_size == Spectral::SegmentSize::UpperHalf) {
      result.get(d) = 0.5 * (result.get(d) + 1.0);
    } else {
      ASSERT(segment_size == Spectral::SegmentSize::Full,
             "Unexpected mortar segment size " << segment_size);
    }
    ++mortar_dimension;
  }
  return result;
}

template <typename TargetFrame>
tnsr::I<DataVector, Dim, TargetFrame> orient_tensor_data_on_slice(
    const tnsr::I<DataVector, Dim, TargetFrame>& tensor,
    const Index<Dim - 1>& slice_extents, const size_t sliced_dimension,
    const OrientationMap<Dim>& orientation) {
  tnsr::I<DataVector, Dim, TargetFrame> result{slice_extents.product()};
  for (size_t d = 0; d < Dim; ++d) {
    result.get(d) = orient_variables_on_slice(tensor.get(d), slice_extents,
                                              sliced_dimension, orientation);
  }
  return result;
}

Variables<tmpl::list<InterfaceTestField>> interface_test_field(
    const ElementId<Dim>& element_id, const std::string& block_name,
    const Mesh<Dim - 1>& face_mesh, const Direction<Dim>& direction) {
  const auto element_logical_coords =
      interface_logical_coordinates(face_mesh, direction);
  const auto element_to_block_map =
      domain::element_to_block_logical_map(element_id);
  const auto block_logical_coords =
      (*element_to_block_map)(element_logical_coords);

  DataVector lambda{};
  if (block_name == "EACylinder" or block_name == "EBCylinder") {
    lambda = 0.5 * (1.0 + block_logical_coords.get(0));
  } else if (block_name == "MAFilledCylinder") {
    lambda = 0.5 * (1.0 - block_logical_coords.get(2));
  } else {
    ASSERT(block_name == "MBFilledCylinder", "Unexpected block " << block_name);
    lambda = 0.5 * (1.0 + block_logical_coords.get(2));
  }

  const DataVector& eta = block_logical_coords.get(1);
  Variables<tmpl::list<InterfaceTestField>> result{
      face_mesh.number_of_grid_points()};
  get(get<InterfaceTestField>(result)) =
      (1.0 + lambda) * (2.0 + sin(eta) + 0.25 * cos(2.0 * eta));
  return result;
}

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

      for (const auto& [direction, neighbors] :
           elements[flattened_element_index].neighbors()) {
        for (const auto& neighbor : neighbors) {
          const auto& neighbor_block = domain.blocks()[neighbor.block_id()];
          if (neighbors.are_conforming()) {
            const auto& neighbor_orientation = neighbors.orientation(neighbor);
            neighbor_meshes[flattened_element_index].emplace(
                DirectionalId{direction, neighbor},
                neighbor_orientation.inverse_map()(
                    ::domain::create_initial_mesh(initial_extents,
                                                  neighbor_block, neighbor,
                                                  i1_basis, i1_quadrature)));
          } else if (elements[flattened_element_index].face_types().at(
                         direction) ==
                     ::domain::FaceType::SingleNonconforming) {
            // We do not insert neighbor meshes into neighbor_mesh for a
            // direction with domain::FaceType::MultipleNonconforming as this
            // could overflow the FixedHashMap size
            neighbor_meshes[flattened_element_index].emplace(
                DirectionalId{direction, neighbor},
                ::domain::create_initial_mesh(initial_extents, neighbor_block,
                                              neighbor, i1_basis,
                                              i1_quadrature));
          }
        }
      }

      flattened_element_index++;
    }
  }

  ASSERT(flattened_element_index == num_elements,
         "flattened_element_index != num_elements");

  const auto& block_names = creator.block_names();
  const auto functions_of_time = creator.functions_of_time();
  const auto find_element_index = [&elements](const ElementId<Dim>& id) {
    const auto element_it = alg::find_if(
        elements,
        [&id](const Element<Dim>& element) { return element.id() == id; });
    ASSERT(element_it != elements.end(), "Could not find element " << id);
    return static_cast<size_t>(std::distance(elements.begin(), element_it));
  };

  // For a conforming interface, discrete reorientation and half-mortar
  // scaling must associate every mortar index with the same physical point.
  // Using orient_variables_on_slice here checks the same point permutation
  // that is used for communicating DG boundary data.
  for (size_t host_index = 0; host_index < elements.size(); ++host_index) {
    const auto& host_element = elements[host_index];
    const auto& host_id = host_element.id();
    const auto& host_name = block_names[host_id.block_id()];
    if (host_name.find("Cylinder") == std::string::npos) {
      continue;
    }

    for (const auto& [host_direction, host_neighbors] :
         host_element.neighbors()) {
      if (not host_neighbors.are_conforming()) {
        continue;
      }
      for (const auto& neighbor_id : host_neighbors) {
        const auto& neighbor_name = block_names[neighbor_id.block_id()];
        if (neighbor_name.find("Cylinder") == std::string::npos) {
          continue;
        }
        CAPTURE(host_name, host_id, host_direction, neighbor_name, neighbor_id);

        const auto& host_to_neighbor_orientation =
            host_neighbors.orientation(neighbor_id);
        const Direction<Dim> neighbor_direction =
            host_to_neighbor_orientation(host_direction.opposite());
        const size_t neighbor_index = find_element_index(neighbor_id);
        const auto& neighbor_element = elements[neighbor_index];
        const auto& neighbor_neighbors =
            neighbor_element.neighbors().at(neighbor_direction);
        REQUIRE(neighbor_neighbors.ids().contains(host_id));
        REQUIRE(neighbor_neighbors.are_conforming());
        const auto& neighbor_to_host_orientation =
            neighbor_neighbors.orientation(host_id);
        REQUIRE(neighbor_to_host_orientation ==
                host_to_neighbor_orientation.inverse_map());

        const DirectionalId<Dim> host_mortar_id{host_direction, neighbor_id};
        const Mesh<Dim - 1> host_face_mesh =
            meshes[host_index].on_interface(host_direction.dimension());
        const Mesh<Dim - 1> host_mortar_mesh = ::dg::mortar_mesh(
            host_face_mesh, neighbor_meshes[host_index]
                                .at(host_mortar_id)
                                .on_interface(host_direction.dimension()));
        const auto host_mortar_size =
            ::dg::mortar_size(host_id, neighbor_id, host_direction.dimension(),
                              host_to_neighbor_orientation);
        const auto host_logical_coords = element_logical_coordinates_on_mortar(
            host_mortar_mesh, host_direction, host_mortar_size);
        const auto host_grid_coords =
            element_maps[host_index](host_logical_coords);

        const DirectionalId<Dim> neighbor_mortar_id{neighbor_direction,
                                                    host_id};
        const Mesh<Dim - 1> neighbor_face_mesh =
            meshes[neighbor_index].on_interface(neighbor_direction.dimension());
        const Mesh<Dim - 1> neighbor_mortar_mesh = ::dg::mortar_mesh(
            neighbor_face_mesh,
            neighbor_meshes[neighbor_index]
                .at(neighbor_mortar_id)
                .on_interface(neighbor_direction.dimension()));
        const auto neighbor_mortar_size = ::dg::mortar_size(
            neighbor_id, host_id, neighbor_direction.dimension(),
            neighbor_to_host_orientation);
        const auto neighbor_logical_coords =
            element_logical_coordinates_on_mortar(
                neighbor_mortar_mesh, neighbor_direction, neighbor_mortar_size);
        const auto neighbor_grid_coords =
            element_maps[neighbor_index](neighbor_logical_coords);

        REQUIRE(host_mortar_mesh ==
                orient_mesh_on_slice(neighbor_mortar_mesh,
                                     neighbor_direction.dimension(),
                                     neighbor_to_host_orientation));
        const auto neighbor_grid_coords_in_host_order =
            orient_tensor_data_on_slice(
                neighbor_grid_coords, neighbor_mortar_mesh.extents(),
                neighbor_direction.dimension(), neighbor_to_host_orientation);
        CHECK_ITERABLE_APPROX(host_grid_coords,
                              neighbor_grid_coords_in_host_order);

        for (const double time : {0.0, 1.0}) {
          CAPTURE(time);
          const auto host_inertial_coords =
              (*grid_to_inertial_maps[host_index])(host_grid_coords, time,
                                                   functions_of_time);
          const auto neighbor_inertial_coords =
              (*grid_to_inertial_maps[neighbor_index])(neighbor_grid_coords,
                                                       time, functions_of_time);
          const auto neighbor_inertial_coords_in_host_order =
              orient_tensor_data_on_slice(
                  neighbor_inertial_coords, neighbor_mortar_mesh.extents(),
                  neighbor_direction.dimension(), neighbor_to_host_orientation);
          CHECK_ITERABLE_APPROX(host_inertial_coords,
                                neighbor_inertial_coords_in_host_order);
        }

        const bool is_m_e_interface = ((host_name == "EACylinder" and
                                        neighbor_name == "MAFilledCylinder") or
                                       (host_name == "MAFilledCylinder" and
                                        neighbor_name == "EACylinder") or
                                       (host_name == "EBCylinder" and
                                        neighbor_name == "MBFilledCylinder") or
                                       (host_name == "MBFilledCylinder" and
                                        neighbor_name == "EBCylinder"));
        if (is_m_e_interface) {
          const auto host_data_on_mortar = ::dg::project_to_mortar(
              interface_test_field(host_id, host_name, host_face_mesh,
                                   host_direction),
              host_face_mesh, host_mortar_mesh, host_mortar_size);
          const auto neighbor_data_on_mortar = ::dg::project_to_mortar(
              interface_test_field(neighbor_id, neighbor_name,
                                   neighbor_face_mesh, neighbor_direction),
              neighbor_face_mesh, neighbor_mortar_mesh, neighbor_mortar_size);
          const auto neighbor_data_in_host_order = orient_variables_on_slice(
              neighbor_data_on_mortar, neighbor_mortar_mesh.extents(),
              neighbor_direction.dimension(), neighbor_to_host_orientation);
          CHECK_ITERABLE_APPROX(
              get(get<InterfaceTestField>(host_data_on_mortar)),
              get(get<InterfaceTestField>(neighbor_data_in_host_order)));
        }
      }
    }
  }
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
      filled_cylinder_extents[1], 4 * filled_cylinder_extents[0] - 3,
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

  // Repeat with z refinement and unequal face extents, so the test exercises
  // multiple neighbors, half mortars, and p projection as in the evolution.
  const size_t production_cylinder_refinement = 1;
  const RefinementMap production_initial_refinement{
      {"CAFilledCylinder", production_cylinder_refinement},
      {"CBFilledCylinder", production_cylinder_refinement},
      {"EAFilledCylinder", production_cylinder_refinement},
      {"EBFilledCylinder", production_cylinder_refinement},
      {"MAFilledCylinder", production_cylinder_refinement},
      {"MBFilledCylinder", production_cylinder_refinement},
      {"CACylinder", production_cylinder_refinement},
      {"CBCylinder", production_cylinder_refinement},
      {"EACylinder", production_cylinder_refinement},
      {"EBCylinder", production_cylinder_refinement},
      {"InnerSphereA", sphere_refinement},
      {"InnerSphereB", sphere_refinement},
      {"OuterSphere", sphere_refinement}};
  const GridPointsMap production_initial_grid_points{
      {"CAFilledCylinder", std::array<size_t, 2>{8, 9}},
      {"CBFilledCylinder", std::array<size_t, 2>{8, 9}},
      {"EAFilledCylinder", std::array<size_t, 2>{8, 9}},
      {"EBFilledCylinder", std::array<size_t, 2>{8, 9}},
      {"MAFilledCylinder", std::array<size_t, 2>{17, 10}},
      {"MBFilledCylinder", std::array<size_t, 2>{17, 10}},
      {"CACylinder", std::array<size_t, 3>{17, 15, 7}},
      {"CBCylinder", std::array<size_t, 3>{17, 15, 7}},
      {"EACylinder", std::array<size_t, 3>{18, 19, 6}},
      {"EBCylinder", std::array<size_t, 3>{18, 19, 6}},
      {"InnerSphereA", inner_sphere_extents},
      {"InnerSphereB", inner_sphere_extents},
      {"OuterSphere", outer_sphere_extents}};
  test(center_a, center_b, radius_a, radius_b, include_sphere_a,
       include_sphere_b, outer_radius, production_initial_refinement,
       production_initial_grid_points, time_dep_options);
}
