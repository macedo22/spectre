// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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

// Tag for the function we want to differentiate: F = 1 / r
template <typename DataType>
struct F : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
Scalar<DataType> get_one_over_radius(const DataType& radius) {
  Scalar<DataType> one_over_r{};
  get(one_over_r) = 1.0 / radius;
  return one_over_r;
}

template <size_t Dim, typename FrameType, typename DataType>
tnsr::i<DataType, Dim, FrameType> get_deriv_one_over_radius(
    const DataType& radius, const tnsr::I<DataType, Dim, FrameType>& x) {
  tnsr::i<DataType, Dim, FrameType> deriv_one_over_radius{};
  for (size_t i = 0; i < Dim; i++) {
    deriv_one_over_radius.get(i) = -x.get(i) / cube(radius);
  }
  return deriv_one_over_radius;
}

template <size_t SpatialDim, typename FrameType, Spectral::Basis Basis,
          Spectral::Quadrature Quadrature>
double get_l2norm(const size_t num_points_1d,
                  const std::array<double, 3>& lower_bound,
                  const std::array<double, 3>& upper_bound) {
  // Setup grid
  Mesh<SpatialDim> mesh{num_points_1d, Spectral::Basis::FiniteDifference,
                        Spectral::Quadrature::CellCentered};
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

  // Compute function to differentiate: F = 1 / r
  const DataVector radius = get(magnitude(x));
  const auto F = get_one_over_radius(radius);

  // Compute expected analytical derivative: d_i F = - x_i / r^3
  const auto analytical_deriv_F = get_deriv_one_over_radius(radius, x);

  // Compute actual numerical derivative of the determinant
  using F_tag = ::F<DataVector>;
  Variables<tmpl::list<F_tag>> F_var(num_points_3d);
  get<F_tag>(F_var) = F;
  const auto numerical_deriv_F_var = partial_derivatives<tmpl::list<F_tag>>(
      F_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& numerical_deriv_F =
      get<Tags::deriv<F_tag, tmpl::size_t<SpatialDim>, FrameType>>(
          numerical_deriv_F_var);

  const double max_diff =
      max(abs(analytical_deriv_F.get(0) - numerical_deriv_F.get(0)));
  std::cout << "Max difference : " << max_diff << std::endl;

  const double l2norm =
      l2Norm(analytical_deriv_F.get(0) - numerical_deriv_F.get(0));
  std::cout << "L2 norm : " << l2norm << std::endl;
  return l2norm;
}

template <Spectral::Basis Basis, Spectral::Quadrature Quadrature>
void test() {
  const size_t dim = 3;
  using frame = Frame::Inertial;

  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.3}};
  const std::array<double, 5> box_lengths{{0.03125, 0.0625, 0.125, 0.25, 0.5}};
  const std::array<size_t, 5> num_points_1d{{4, 6, 8, 10, 12}};
  // Table of L2 norms of difference between analytical and numerical
  // derivatives for each box length and # 1D grid points
  std::array<std::array<double, box_lengths.size()>, num_points_1d.size()>
      l2norms{};

  std::cout << "==== L2 norm ====" << std::endl;
  for (size_t i = 0; i < num_points_1d.size(); i++) {
    std::cout << "Num points 1D : " << num_points_1d[i] << std::endl
              << std::endl;
    for (size_t n = 0; n < box_lengths.size(); n++) {
      const double length = box_lengths[n];
      const std::array<double, 3> upper_bound{{lower_bound[0] + length,
                                               lower_bound[1] + length,
                                               lower_bound[2] + length}};
      std::cout << "Length : " << length << std::endl;
      std::cout << "lower bound : " << lower_bound << std::endl;
      std::cout << "upper bound : " << upper_bound << std::endl;
      const double l2norm = get_l2norm<dim, frame, Basis, Quadrature>(
          num_points_1d[i], lower_bound, upper_bound);
      l2norms[i][n] = l2norm;
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "====================================================="
            << std::endl;

  std::array<std::array<double, box_lengths.size() - 1>, num_points_1d.size()>
      convergence_orders{};
  std::cout << "\n==== L2 norm convergence orders ====" << std::endl;
  for (size_t i = 0; i < num_points_1d.size(); i++) {
    std::cout << "Num points 1D : " << num_points_1d[i] << std::endl
              << std::endl;

    for (size_t n = 1; n < box_lengths.size(); n++) {
      const double length = box_lengths[n];
      const double half_length = box_lengths[n - 1];
      const std::array<double, 3> upper_bound{{lower_bound[0] + length,
                                               lower_bound[1] + length,
                                               lower_bound[2] + length}};
      std::cout << "Length : " << length << std::endl;
      std::cout << "lower bound : " << lower_bound << std::endl;
      std::cout << "upper bound : " << upper_bound << std::endl;
      std::cout << "Half length : " << half_length << std::endl;

      const double l2norm = l2norms[i][n];
      const double l2_norm_with_half_length = l2norms[i][n - 1];
      const double convergence_order = log2(l2norm / l2_norm_with_half_length);
      convergence_orders[i][n - 1] = convergence_order;
      std::cout << "Convergence order : " << convergence_order << std::endl;
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "====================================================="
            << std::endl;
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Gr.NumericalDerivative",
    "[PointwiseFunctions][Unit]") {
  std::cout << "=== LEGENDRE / GAUSSLOBATTO ===\n" << std::endl;
  test<Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>();
  std::cout << "\n\n\n=== FINITEDIFFERENCE / CELLCENTERED ===\n" << std::endl;
  test<Spectral::Basis::FiniteDifference, Spectral::Quadrature::CellCentered>();
}
