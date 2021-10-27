// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
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
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <typename DataType>
struct H : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
Scalar<DataType> get_H(const double mass, const DataType& radius) {
  Scalar<DataType> H{};
  get(H) = mass / radius;
  return H;
}

template <size_t Dim, typename FrameType, typename DataType>
tnsr::i<DataType, Dim, FrameType> get_deriv_H(
    const double mass, const DataType& radius,
    const tnsr::I<DataType, Dim, FrameType>& x) {
  tnsr::i<DataType, Dim, FrameType> deriv_H{};
  for (size_t i = 0; i < Dim; i++) {
    deriv_H.get(i) = -mass * x.get(i) / cube(radius);
  }
  return deriv_H;
}

template <size_t SpatialDim, typename FrameType>
void test_numerical_deriv_H(const double mass) {
  // Parameters for KerrSchild solution
  const size_t num_points_1d = 8;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.3}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};

  // Setup grid
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

  // Compute our function for non-spinning BH, H = M / r
  const DataVector r = get(magnitude(x));
  const auto H = get_H(mass, r);

  // Compute expected analytical derivative of the determinant
  const auto analytical_deriv_H = get_deriv_H(mass, r, x);

  // Compute actual numerical derivative of the determinant
  using H_tag = ::H<DataVector>;
  Variables<tmpl::list<H_tag>> H_var(num_points_3d);
  get<H_tag>(H_var) = H;
  const auto numerical_deriv_H_var = partial_derivatives<tmpl::list<H_tag>>(
      H_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& numerical_deriv_H =
      get<Tags::deriv<H_tag, tmpl::size_t<SpatialDim>, FrameType>>(
          numerical_deriv_H_var);

  // Best is 1e-12
  Approx approx = Approx::custom().epsilon(1e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(analytical_deriv_H, numerical_deriv_H, approx);
}

template <size_t SpatialDim, typename FrameType>
void test_numerical_deriv_H_l2_norm(const double mass,
                                    const size_t num_points_1d,
                                    const std::array<double, 3>& lower_bound,
                                    const std::array<double, 3>& upper_bound) {
  // Setup grid
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

  // Compute our function for non-spinning BH, H = M / r
  const DataVector r = get(magnitude(x));
  const auto H = get_H(mass, r);

  // Compute expected analytical derivative of the determinant
  const auto analytical_deriv_H = get_deriv_H(mass, r, x);

  // Compute actual numerical derivative of the determinant
  using H_tag = ::H<DataVector>;
  Variables<tmpl::list<H_tag>> H_var(num_points_3d);
  get<H_tag>(H_var) = H;
  const auto numerical_deriv_H_var = partial_derivatives<tmpl::list<H_tag>>(
      H_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& numerical_deriv_H =
      get<Tags::deriv<H_tag, tmpl::size_t<SpatialDim>, FrameType>>(
          numerical_deriv_H_var);

  const double l2norm =
      l2Norm(analytical_deriv_H.get(0) - numerical_deriv_H.get(0));
  std::cout << "L2 norm : " << l2norm << std::endl;
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Gr.NumericalDerivative",
    "[PointwiseFunctions][Unit]") {
  const double mass = 1.01;
  const size_t dim = 3;
  using frame = Frame::Inertial;

  // Test numerical derivative of H against analytical derivative
  test_numerical_deriv_H<dim, frame>(mass);

  // Print L2 norms of difference between analytical and numerical derivatives
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.3}};
  const std::array<double, 5> box_lengths{{0.01, 0.05, 0.1, 0.25, 0.5}};
  const std::array<size_t, 5> num_points_1d{{4, 6, 8, 10, 12}};
  std::cout << "==== KerrSchild deriv_H L2 norm ====" << std::endl;
  for (size_t n = 0; n < box_lengths.size(); n++) {
    const double length = box_lengths[n];
    const std::array<double, 3> upper_bound{{lower_bound[0] + length,
                                             lower_bound[1] + length,
                                             lower_bound[2] + length}};
    std::cout << "lower bound : " << lower_bound << std::endl;
    std::cout << "upper bound : " << upper_bound << std::endl << std::endl;
    for (size_t i = 0; i < num_points_1d.size(); i++) {
      std::cout << "Num points 1D : " << num_points_1d[i] << std::endl;
      test_numerical_deriv_H_l2_norm<dim, frame>(mass, num_points_1d[i],
                                                 lower_bound, upper_bound);
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "====================================================="
            << std::endl;
}
