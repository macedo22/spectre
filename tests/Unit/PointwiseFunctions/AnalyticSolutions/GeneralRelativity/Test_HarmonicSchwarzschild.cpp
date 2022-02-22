// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.32;
  get<1>(x) = 0.82;
  get<2>(x) = 1.24;
  return x;
}

template <typename Frame, typename DataType>
void test_tag_retrieval(const DataType& used_for_size) {
  // Parameters for HarmonicSchwarzschild solution
  const double mass = 1.234;
  const std::array<double, 3> center{{1.0, 2.0, 3.0}};
  const auto x = spatial_coords<Frame>(used_for_size);
  const double t = 1.3;

  // Evaluate solution
  const gr::Solutions::HarmonicSchwarzschild solution(mass, center);
  TestHelpers::AnalyticSolutions::test_tag_retrieval(
      solution, x, t,
      typename gr::Solutions::HarmonicSchwarzschild::template tags<DataType,
                                                                   Frame>{});
}

void test_serialize() {
  gr::Solutions::HarmonicSchwarzschild solution(3.0, {{0.0, 3.0, 4.0}});
  test_serialization(solution);
}

void test_copy_and_move() {
  gr::Solutions::HarmonicSchwarzschild solution(3.0, {{0.0, 3.0, 4.0}});
  test_copy_semantics(solution);
  auto solution_copy = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}

void test_construct_from_options() {
  const auto created =
      TestHelpers::test_creation<gr::Solutions::HarmonicSchwarzschild>(
          "Mass: 0.5\n"
          "Center: [1.0,3.0,2.0]");
  CHECK(created ==
        gr::Solutions::HarmonicSchwarzschild(0.5, {{1.0, 3.0, 2.0}}));
}

template <typename FrameType, typename DataType>
void test_computed_quantities(const DataType /*used_for_size*/) {
  // Parameters for HarmonicSchwarzschild solution
  const double mass = 1.01;
  const std::array<double, 3> center{{0.2, -0.1, 0.4}};
  gr::Solutions::HarmonicSchwarzschild solution(mass, center);

  // Setup grid
  const size_t num_points_1d = 8;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{num_points_1d, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  const size_t num_points_3d = num_points_1d * num_points_1d * num_points_1d;
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();
  const auto vars = solution.variables(
      x, t,
      typename gr::Solutions::HarmonicSchwarzschild::tags<DataVector,
                                                          FrameType>{});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Gr.HarmonicSchwarzschild",
    "[PointwiseFunctions][Unit]") {
  test_copy_and_move();
  test_serialize();
  test_construct_from_options();

  test_tag_retrieval<Frame::Inertial>(DataVector(5));
  test_tag_retrieval<Frame::Inertial>(0.0);

  test_tag_retrieval<Frame::Grid>(DataVector(5));
  test_tag_retrieval<Frame::Grid>(0.0);

  test_computed_quantities<Frame::Inertial>(DataVector(5));
  test_computed_quantities<Frame::Inertial>(0.0);

  test_computed_quantities<Frame::Grid>(DataVector(5));
  test_computed_quantities<Frame::Grid>(0.0);
}

// TODO put back OutputRegex error tests
