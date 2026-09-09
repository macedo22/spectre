// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;

::domain::CoordinateMap<
    Frame::BlockLogical, Frame::Inertial,
    domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>>
make_affine_map_3d(const std::array<double, 3>& center,
                   const std::array<double, 3>& dimensions) {
  return domain::make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>{
          Affine{-1.0, 1.0, center[0] - 0.5 * dimensions[0],
                 center[0] + 0.5 * dimensions[0]},
          Affine{-1.0, 1.0, center[1] - 0.5 * dimensions[1],
                 center[1] + 0.5 * dimensions[1]},
          Affine{-1.0, 1.0, center[2] - 0.5 * dimensions[2],
                 center[2] + 0.5 * dimensions[2]}});
}

struct ConformingCubes {
  static constexpr size_t Dim = 3;

  ConformingCubes() {}
  Domain<Dim> create_domain() const {
    const std::array<double, Dim> center_block_1{-0.8, 1.3, 4.1};
    const std::array<double, Dim> dimensions_block_1{5.0, 6.0, 7.0};

    auto coord_map_block_1 =
        make_affine_map_3d(center_block_1, dimensions_block_1);

    // block 2 has a shift in the x coord because it abuts block 1 on +x side
    const std::array<double, Dim> center_block_2{
        center_block_1[0] + dimensions_block_1[0], center_block_1[1],
        center_block_1[2]};
    const std::array<double, Dim> dimensions_block_2 = dimensions_block_1;

    // [-1, 1]^3
    auto unit_cube = make_affine_map_3d(std::array<double, Dim>{0.0, 0.0, 0.0},
                                        std::array<double, Dim>{2.0, 2.0, 2.0});
    const OrientationMap<3> rotation_block_2{std::array<Direction<Dim>, Dim>{
        Direction<3>::lower_zeta(), Direction<Dim>::lower_xi(),
        Direction<3>::upper_eta()}};
    // rotate unit cube, then translate and scale it to abut block 1
    auto coord_map_block_2 = domain::push_back(
        domain::push_back(
            unit_cube,
            domain::CoordinateMaps::DiscreteRotation<Dim>(rotation_block_2)),
        make_affine_map_3d(center_block_2, dimensions_block_2));

    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, Dim>>>
        coordinate_maps{};
    coordinate_maps.emplace_back(
        std::make_unique<std::decay_t<decltype(coord_map_block_1)>>(
            std::move(coord_map_block_1)));
    coordinate_maps.emplace_back(
        std::make_unique<std::decay_t<decltype(coord_map_block_2)>>(
            std::move(coord_map_block_2)));

    std::vector<DirectionMap<Dim, BlockNeighbors<Dim>>> block_neighbors{
        coordinate_maps.size()};
    // add block 2 as a neighbor of block 1
    block_neighbors[0].emplace(Direction<Dim>::upper_xi(),
                               BlockNeighbors<Dim>{{1},
                                                   {{1, rotation_block_2}},
                                                   /*are_conforming=*/true});
    // add block 1 as a neighbor of block 2
    block_neighbors[1].emplace(
        Direction<Dim>::upper_zeta(),
        BlockNeighbors<Dim>{{0},
                            {{0, rotation_block_2.inverse_map()}},
                            /*are_conforming=*/true});

    std::vector<Block<Dim>> blocks;
    blocks.reserve(coordinate_maps.size());

    blocks.emplace_back(std::move(coordinate_maps[0]), 0,
                        std::move(block_neighbors[0]), block_names_.at(0),
                        domain::topologies::hypercube<Dim>);
    blocks.emplace_back(std::move(coordinate_maps[1]), 1,
                        std::move(block_neighbors[1]), block_names_.at(1),
                        domain::topologies::hypercube<Dim>);

    Domain<3> domain{std::move(blocks), {}, block_groups};

    return domain;
  }

  std::vector<std::string> block_names_{"Block1", "Block2"};
  std::unordered_map<std::string, std::unordered_set<std::string>> block_groups{
      {"Blocks", {{"Block1", "Block2"}}}};
};

// inner_radius = 0.0 for a cylinder, inner_radius > 0.0 for a hollow cylinder
domain::CoordinateMap<
    Frame::BlockLogical, Frame::Inertial,
    domain::CoordinateMaps::ProductOf3Maps<
        ::domain::CoordinateMaps::Affine, ::domain::CoordinateMaps::Identity<1>,
        ::domain::CoordinateMaps::Interval>,
    domain::CoordinateMaps::ProductOf2Maps<
        ::domain::CoordinateMaps::PolarToCartesian,
        ::domain::CoordinateMaps::Identity<1>>>
make_cyl_coordinate_map(const double inner_radius, const double outer_radius,
                   const double lower_z_bound, const double upper_z_bound) {
  using Affine = domain::CoordinateMaps::Affine;
  using Identity1D = domain::CoordinateMaps::Identity<1>;
  using Interval = domain::CoordinateMaps::Interval;
  using PolarToCartesian = domain::CoordinateMaps::PolarToCartesian;
  const auto linear = domain::CoordinateMaps::Distribution::Linear;

  // Map: (xi, eta, zeta) in [-1,1] x [0, 2pi] x [-1, 1]
  //   xi -> r in [inner_r, outer_r]  (Affine)
  //   eta -> phi in [0, 2pi)         (Identity<1>, passes through)
  //   zeta -> z in [z_lower, z_upper] (Interval)
  // Then PolarToCartesian x Identity<1> maps (r, phi, z) -> (x, y, z)
  return domain::make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
      domain::CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Interval>{
          Affine{-1.0, 1.0, inner_radius, outer_radius}, Identity1D{},
          Interval{-1.0, 1.0, lower_z_bound, upper_z_bound, linear}},
      domain::CoordinateMaps::ProductOf2Maps<PolarToCartesian, Identity1D>{
          PolarToCartesian{}, Identity1D{}});
}

struct ConformingReversedCylinders {
  static constexpr size_t Dim = 3;

  ConformingReversedCylinders() {}
  Domain<Dim> create_domain() const {
    const double radius =  5.0;
    const double height = 8.0;

    // const double unit_inner_radius = 0.0;
    // const double unit_outer_radius = 1.0;
    // const double unit_lower_bound_z = -1.0;
    // const double unit_upper_bound_z = 1.0;

    // const double unit_inner_radius = 0.0;
    // const double unit_outer_radius = 1.0;
    // const double unit_lower_bound_z = -1.0;
    // const double unit_upper_bound_z = 1.0;

    // const auto logical_to_unit_cylinder_map =
    //   cyl_coordinate_map(unit_inner_radius, unit_outer_radius,
    //                      unit_lower_bound_z, unit_upper_bound_z);

    // auto coord_map_block_1 =
    //     make_affine_map_3d(center_block_1, dimensions_block_1);

    auto coord_map_block_1 = make_affine_map_3d(0.0, radius, 0.0, height);

    // // block 2 has a shift in the x coord because it abuts block 1 on +x side
    // const std::array<double, Dim> center_block_2{
    //     center_block_1[0] + dimensions_block_1[0], center_block_1[1],
    //     center_block_1[2]};
    // const std::array<double, Dim> dimensions_block_2 = dimensions_block_1;

    // // [-1, 1]^3
    // auto unit_cube = make_affine_map_3d(std::array<double, Dim>{0.0, 0.0, 0.0},
    //                                     std::array<double, Dim>{2.0, 2.0, 2.0});
    // const OrientationMap<3> rotation_block_2{std::array<Direction<Dim>, Dim>{
    //     Direction<3>::lower_zeta(), Direction<Dim>::lower_xi(),
    //     Direction<3>::upper_eta()}};
    // // rotate unit cube, then translate and scale it to abut block 1
    // auto coord_map_block_2 = domain::push_back(
    //     domain::push_back(
    //         unit_cube,
    //         domain::CoordinateMaps::DiscreteRotation<Dim>(rotation_block_2)),
    //     make_affine_map_3d(center_block_2, dimensions_block_2));

  const OrientationMap<Dim> rotate_upside_down{std::array<Direction<Dim>, Dim>{
      Direction<3>::lower_xi(), Direction<3>::upper_eta(),
      Direction<3>::lower_zeta()}};

    auto coord_map_block_2 = domain::push_back(
            coord_map_block_1,
            domain::CoordinateMaps::DiscreteRotation<Dim>(rotate_upside_down));

    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, Dim>>>
        coordinate_maps{};
    coordinate_maps.emplace_back(
        std::make_unique<std::decay_t<decltype(coord_map_block_1)>>(
            std::move(coord_map_block_1)));
    coordinate_maps.emplace_back(
        std::make_unique<std::decay_t<decltype(coord_map_block_2)>>(
            std::move(coord_map_block_2)));

    std::vector<DirectionMap<Dim, BlockNeighbors<Dim>>> block_neighbors{
        coordinate_maps.size()};
    // add block 2 as a neighbor of block 1
    block_neighbors[0].emplace(Direction<Dim>::upper_xi(),
                               BlockNeighbors<Dim>{{1},
                                                   {{1, rotation_block_2}},
                                                   /*are_conforming=*/true});
    // add block 1 as a neighbor of block 2
    block_neighbors[1].emplace(
        Direction<Dim>::upper_zeta(),
        BlockNeighbors<Dim>{{0},
                            {{0, rotation_block_2.inverse_map()}},
                            /*are_conforming=*/true});

    std::vector<Block<Dim>> blocks;
    blocks.reserve(coordinate_maps.size());

    blocks.emplace_back(std::move(coordinate_maps[0]), 0,
                        std::move(block_neighbors[0]), block_names_.at(0),
                        domain::topologies::hypercube<Dim>);
    blocks.emplace_back(std::move(coordinate_maps[1]), 1,
                        std::move(block_neighbors[1]), block_names_.at(1),
                        domain::topologies::hypercube<Dim>);

    Domain<3> domain{std::move(blocks), {}, block_groups};

    return domain;
  }

  std::vector<std::string> block_names_{"Block1", "Block2"};
  std::unordered_map<std::string, std::unordered_set<std::string>> block_groups{
      {"Blocks", {{"Block1", "Block2"}}}};
};

template <typename DataType, size_t SpatialDim>
tnsr::II<DataType, SpatialDim> random_inv_spatial_metric(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(-0.05, 0.05);
  auto inv_spatial_metric =
      make_with_random_values<tnsr::II<DataType, SpatialDim>>(
          generator, make_not_null(&distribution), used_for_size);
  for (size_t d = 0; d < SpatialDim; ++d) {
    inv_spatial_metric.get(d, d) += 1.0;
  }
  return inv_spatial_metric;
}

template <size_t SpatialDim, typename DataType>
void test_euclidean_basis_vector(const DataType& used_for_size) {
  for (const auto& direction : Direction<SpatialDim>::all_directions()) {
    auto expected =
        make_with_value<tnsr::i<DataType, SpatialDim>>(used_for_size, 0.0);
    expected.get(direction.axis()) =
        make_with_value<DataType>(used_for_size, direction.sign());

    CHECK_ITERABLE_APPROX((euclidean_basis_vector(direction, used_for_size)),
                          std::move(expected));
  }
}

template <size_t SpatialDim, typename DataType>
void test_unit_basis_form(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  const auto inv_spatial_metric =
      random_inv_spatial_metric<DataType, SpatialDim>(make_not_null(&generator),
                                                      used_for_size);
  for (const auto& direction : Direction<SpatialDim>::all_directions()) {
    const auto basis_form = unit_basis_form(direction, inv_spatial_metric);
    auto expected = euclidean_basis_vector(direction, used_for_size);
    const DataType norm = get(magnitude(expected, inv_spatial_metric));
    for (size_t d = 0; d < SpatialDim; ++d) {
      expected.get(d) /= norm;
    }
    CHECK_ITERABLE_APPROX(basis_form, expected);
    CHECK_ITERABLE_APPROX(get(magnitude(expected, inv_spatial_metric)),
                          make_with_value<DataType>(used_for_size, 1.0));
  }
}

void test_physical_separation() {
  const ConformingCubes conforming_cubes_creator{};
  const Domain<3> conforming_cubes_domain =
      conforming_cubes_creator.create_domain();
  const auto& conforming_cubes_blocks = conforming_cubes_domain.blocks();
  test_physical_separation(conforming_cubes_blocks, 0.0);
}
}  //  namespace

SPECTRE_TEST_CASE("Unit.Domain.TestHelpers", "[Unit][Domain]") {
  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_euclidean_basis_vector, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_unit_basis_form, (1, 2, 3));

  test_physical_separation();
}
