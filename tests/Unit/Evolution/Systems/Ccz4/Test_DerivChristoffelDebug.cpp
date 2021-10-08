// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/Ricci.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

Scalar<DataVector> get_conformal_factor(
    const Scalar<DataVector>& det_spatial_metric) {
  return Scalar<DataVector>{pow(get(det_spatial_metric), -1. / 6.)};
}

template <size_t Dim, typename Frame>
tnsr::ii<DataVector, Dim, Frame> get_conformal_spatial_metric(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, Dim, Frame>& spatial_metric) {
  tnsr::ii<DataVector, Dim, Frame> result{};
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = i; j < Dim; j++) {
      result.get(i, j) = get(conformal_factor) * get(conformal_factor) *
                         spatial_metric.get(i, j);
    }
  }
  return result;
}

template <size_t Dim, typename Frame>
tnsr::iJJ<DataVector, Dim, Frame> get_field_d_up(
    const tnsr::ijj<DataVector, Dim, Frame>& field_d,
    const tnsr::II<DataVector, Dim, Frame>& inverse_conformal_spatial_metric) {
  auto result = make_with_value<tnsr::iJJ<DataVector, Dim, Frame>>(
      get<0, 0, 0>(field_d), 0.0);
  for (size_t k = 0; k < Dim; k++) {
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = i; j < Dim; j++) {
        for (size_t m = 0; m < Dim; m++) {
          for (size_t n = 0; n < Dim; n++) {
            result.get(k, i, j) += inverse_conformal_spatial_metric.get(i, n) *
                                   inverse_conformal_spatial_metric.get(m, j) *
                                   field_d.get(k, n, m);
          }
        }
      }
    }
  }
  return result;
}

template <typename Solution>
void test_compute(const Solution& solution, size_t grid_size_each_dimension,
                  const std::array<double, 3>& lower_bound,
                  const std::array<double, 3>& upper_bound) {
  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });
  const size_t num_points_3d = grid_size_each_dimension *
                               grid_size_each_dimension *
                               grid_size_each_dimension;
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();
  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<SpatialDim>>(vars);
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  //   std::cout << "det_spatial_metric : " << det_spatial_metric << std::endl;
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  //   const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);
  const auto conformal_factor = get_conformal_factor(det_spatial_metric);
  //   std::cout << "conformal_factor : " << conformal_factor << std::endl;

  const auto conformal_spatial_metric =
      get_conformal_spatial_metric(conformal_factor, spatial_metric);

  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(conformal_spatial_metric).second;

  using conformal_spatial_metric_tag =
      Ccz4::Tags::ConformalMetric<SpatialDim, Frame::Inertial, DataVector>;
  Variables<tmpl::list<conformal_spatial_metric_tag>>
      conformal_spatial_metric_var(num_points_3d);
  get<conformal_spatial_metric_tag>(conformal_spatial_metric_var) =
      conformal_spatial_metric;
  const auto d_conformal_spatial_metric_var =
      partial_derivatives<tmpl::list<conformal_spatial_metric_tag>>(
          conformal_spatial_metric_var, mesh,
          coord_map.inv_jacobian(x_logical));
  const auto& d_conformal_spatial_metric =
      get<Tags::deriv<conformal_spatial_metric_tag, tmpl::size_t<SpatialDim>,
                      Frame::Inertial>>(d_conformal_spatial_metric_var);

  tnsr::ijj<DataVector, SpatialDim, Frame::Inertial> field_d{};
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        field_d.get(k, i, j) = 0.5 * d_conformal_spatial_metric.get(k, i, j);
      }
    }
  }

  // inverse conformal spatial metric * D
  const auto expected_constraint_1 =
      make_with_value<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          get<0, 0, 0>(field_d), 0.0);

  // inverse conformal spatial metric * D
  auto constraint_1 =
      make_with_value<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          get<0, 0, 0>(field_d), 0.0);
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = 0; j < SpatialDim; j++) {
        constraint_1.get(k) +=
            inverse_conformal_spatial_metric.get(i, j) * field_d.get(k, i, j);
      }
    }
  }
  //   CHECK_ITERABLE_APPROX(expected_constraint_1, constraint_1);

  const auto field_d_up =
      get_field_d_up(field_d, inverse_conformal_spatial_metric);

  const DataVector used_for_size =
      DataVector(num_points_3d, std::numeric_limits<double>::signaling_NaN());
  auto expected_field_d_up =
      make_with_value<tnsr::iJJ<DataVector, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        for (size_t n = 0; n < SpatialDim; n++) {
          for (size_t m = 0; m < SpatialDim; m++) {
            expected_field_d_up.get(k, i, j) +=
                0.5 * inverse_conformal_spatial_metric.get(i, n) *
                inverse_conformal_spatial_metric.get(m, j) *
                d_conformal_spatial_metric.get(k, n, m);
          }
        }
      }
    }
  }

  // passes
  CHECK_ITERABLE_APPROX(expected_field_d_up, field_d_up);

  using field_d_tag =
      Ccz4::Tags::FieldD<SpatialDim, Frame::Inertial, DataVector>;
  Variables<tmpl::list<field_d_tag>> field_d_var(num_points_3d);
  get<field_d_tag>(field_d_var) = field_d;
  const auto d_field_d_var = partial_derivatives<tmpl::list<field_d_tag>>(
      field_d_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_field_d =
      get<Tags::deriv<field_d_tag, tmpl::size_t<SpatialDim>, Frame::Inertial>>(
          d_field_d_var);

  const auto d_conformal_christoffel_second_kind =
      Ccz4::deriv_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d, d_field_d, field_d_up);

  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);

  using conformal_factor_tag = Ccz4::Tags::ConformalFactor<DataVector>;
  Variables<tmpl::list<conformal_factor_tag>> conformal_factor_var(
      num_points_3d);
  get<conformal_factor_tag>(conformal_factor_var) = conformal_factor;
  const auto d_conformal_factor_var =
      partial_derivatives<tmpl::list<conformal_factor_tag>>(
          conformal_factor_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_conformal_factor =
      get<Tags::deriv<conformal_factor_tag, tmpl::size_t<SpatialDim>,
                      Frame::Inertial>>(d_conformal_factor_var);

  tnsr::i<DataVector, SpatialDim, Frame::Inertial> field_p{};
  for (size_t i = 0; i < SpatialDim; i++) {
    field_p.get(i) = d_conformal_factor.get(i) / get(conformal_factor);
  }

  using field_p_tag =
      Ccz4::Tags::FieldP<SpatialDim, Frame::Inertial, DataVector>;
  Variables<tmpl::list<field_p_tag>> field_p_var(num_points_3d);
  get<field_p_tag>(field_p_var) = field_p;
  const auto d_field_p_var = partial_derivatives<tmpl::list<field_p_tag>>(
      field_p_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_field_p =
      get<Tags::deriv<field_p_tag, tmpl::size_t<SpatialDim>, Frame::Inertial>>(
          d_field_p_var);

  using conformal_christoffel_second_kind_tag =
      Ccz4::Tags::ConformalChristoffelSecondKind<SpatialDim, Frame::Inertial,
                                                 DataVector>;
  Variables<tmpl::list<conformal_christoffel_second_kind_tag>>
      conformal_christoffel_second_kind_var(num_points_3d);
  get<conformal_christoffel_second_kind_tag>(
      conformal_christoffel_second_kind_var) =
      conformal_christoffel_second_kind;
  const auto d_conformal_christoffel_second_kind_var =
      partial_derivatives<tmpl::list<conformal_christoffel_second_kind_tag>>(
          conformal_christoffel_second_kind_var, mesh,
          coord_map.inv_jacobian(x_logical));
  const auto& expected_d_conformal_christoffel_second_kind =
      get<Tags::deriv<conformal_christoffel_second_kind_tag,
                      tmpl::size_t<SpatialDim>, Frame::Inertial>>(
          d_conformal_christoffel_second_kind_var);

  //   CHECK_ITERABLE_APPROX(expected_d_conformal_christoffel_second_kind,
  //   d_conformal_christoffel_second_kind);
  // passes with 1e-11 at best when deriv_conformal_christoffel is updated to
  // not symmetrize
  Approx approx1 = Approx::custom().epsilon(1e-14).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_d_conformal_christoffel_second_kind,
                               d_conformal_christoffel_second_kind, approx1);

  const auto christoffel_second_kind = Ccz4::christoffel_second_kind(
      conformal_spatial_metric, inverse_conformal_spatial_metric, field_p,
      conformal_christoffel_second_kind);

  using christoffel_second_kind_tag =
      gr::Tags::SpatialChristoffelSecondKind<SpatialDim, Frame::Inertial,
                                             DataVector>;
  Variables<tmpl::list<christoffel_second_kind_tag>>
      christoffel_second_kind_var(num_points_3d);
  get<christoffel_second_kind_tag>(christoffel_second_kind_var) =
      christoffel_second_kind;
  const auto d_christoffel_second_kind_var =
      partial_derivatives<tmpl::list<christoffel_second_kind_tag>>(
          christoffel_second_kind_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_christoffel_second_kind =
      get<Tags::deriv<christoffel_second_kind_tag, tmpl::size_t<SpatialDim>,
                      Frame::Inertial>>(d_christoffel_second_kind_var);

  // Compute expected and actual ricci tensors using above computed arguments
  const auto expected_python_ricci_tensor{
      pypp::call<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          "Ricci", "spatial_ricci_tensor", christoffel_second_kind,
          d_conformal_christoffel_second_kind, conformal_spatial_metric,
          inverse_conformal_spatial_metric, field_d, field_d_up, field_p,
          d_field_p)};

  const auto expected_cpp_gr_ricci_tensor =
      gr::ricci_tensor(christoffel_second_kind, d_christoffel_second_kind);

  const auto actual_ricci_tensor = Ccz4::spatial_ricci_tensor(
      christoffel_second_kind, d_conformal_christoffel_second_kind,
      conformal_spatial_metric, inverse_conformal_spatial_metric, field_d,
      field_d_up, field_p, d_field_p);

  CHECK_ITERABLE_APPROX(expected_python_ricci_tensor, actual_ricci_tensor);

  Approx approx = Approx::custom().epsilon(1e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_cpp_gr_ricci_tensor,
                               actual_ricci_tensor, approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.DerivChristoffelDebug",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  const double mass = 2.;
  const std::array<double, 3> spin{{0.3, 0.5, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 8;
  //   const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> lower_bound{{8.2, 12.4, 13.2}};
  const std::array<double, 3> upper_bound{{8.0, 12.2, 13.0}};

  test_compute(solution, grid_size, lower_bound, upper_bound);
}
