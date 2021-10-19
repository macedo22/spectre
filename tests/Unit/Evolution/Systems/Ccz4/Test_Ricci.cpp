// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <climits>
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

template <typename Solution>
void test_compute_spatial_ricci_tensor(
    const Solution& solution, size_t grid_size_each_dimension,
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
  const auto& d_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<SpatialDim>,
                      tmpl::size_t<SpatialDim>, Frame::Inertial>>(vars);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  // Compute arguments for `spatial_ricci_tensor` function to test
  const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);
  // TODO: try thisL
  //   auto conformal_factor = pow(get(det_spatial_metric), 1. / 6.);
  //   get(conformal_factor) = 1.0 / get(conformal_factor);

  // TODO: compute partial deriv of conformal metric analytically

  // TODO: see where precision starts falling off, try different O.O.O.

  //   std::cout << "conformal_factor : " << conformal_factor << std::endl;

  tnsr::ii<DataVector, SpatialDim, Frame::Inertial> conformal_spatial_metric{};
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = i; j < SpatialDim; j++) {
      conformal_spatial_metric.get(i, j) =
          square(conformal_factor) * spatial_metric.get(i, j);
    }
  }

  //   const auto det_conformal_spatial_metric =
  //       determinant_and_inverse(conformal_spatial_metric).first;
  // okay: is 1
  //   std::cout << "det_conformal_spatial_metric : " <<
  //   det_conformal_spatial_metric << std::endl;
  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(conformal_spatial_metric).second;
  //   tnsr::II<DataVector, SpatialDim, Frame::Inertial>
  //       expected_inverse_conformal_spatial_metric{};
  //   for (size_t i = 0; i < SpatialDim; i++) {
  //     for (size_t j = i; j < SpatialDim; j++) {
  //       expected_inverse_conformal_spatial_metric.get(i, j) =
  //           inverse_spatial_metric.get(i, j) / pow<2>(conformal_factor);
  //     }
  //   }

  // passes
  //   CHECK_ITERABLE_APPROX(expected_inverse_conformal_spatial_metric,
  //                         inverse_conformal_spatial_metric);

  //   using conformal_spatial_metric_tag =
  //       Ccz4::Tags::ConformalMetric<SpatialDim, Frame::Inertial, DataVector>;
  //   Variables<tmpl::list<conformal_spatial_metric_tag>>
  //       conformal_spatial_metric_var(num_points_3d);
  //   get<conformal_spatial_metric_tag>(conformal_spatial_metric_var) =
  //       conformal_spatial_metric;
  //   const auto d_conformal_spatial_metric_var =
  //       partial_derivatives<tmpl::list<conformal_spatial_metric_tag>>(
  //           conformal_spatial_metric_var, mesh,
  //           coord_map.inv_jacobian(x_logical));
  //   const auto& d_conformal_spatial_metric =
  //       get<Tags::deriv<conformal_spatial_metric_tag,
  //       tmpl::size_t<SpatialDim>,
  //                       Frame::Inertial>>(d_conformal_spatial_metric_var);

  gr::Solutions::KerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      ks_cache(solution, x);
  const auto d_det_spatial_metric = ks_cache.get_var(
      gr::Tags::DerivDetSpatialMetric<SpatialDim, Frame::Inertial,
                                      DataVector>{});

  tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>
      d_conformal_spatial_metric{};
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        d_conformal_spatial_metric.get(k, i, j) =
            pow<2>(conformal_factor) * d_spatial_metric.get(k, i, j) -
            pow<8>(conformal_factor) * d_det_spatial_metric.get(k) *
                spatial_metric.get(i, j) / 3.;
      }
    }
  }

  //   using conformal_factor_tag = Ccz4::Tags::ConformalFactor<DataVector>;
  //   Variables<tmpl::list<conformal_factor_tag>> conformal_factor_var(
  //       num_points_3d);
  //   get(get<conformal_factor_tag>(conformal_factor_var)) = conformal_factor;
  //   const auto d_conformal_factor_var =
  //       partial_derivatives<tmpl::list<conformal_factor_tag>>(
  //           conformal_factor_var, mesh, coord_map.inv_jacobian(x_logical));
  //   const auto& d_conformal_factor =
  //       get<Tags::deriv<conformal_factor_tag, tmpl::size_t<SpatialDim>,
  //                       Frame::Inertial>>(d_conformal_factor_var);

  // just use conformal factor tag for phi^2
  //   using conformal_factor_tag = Ccz4::Tags::ConformalFactor<DataVector>;
  //   Variables<tmpl::list<conformal_factor_tag>> conformal_factor_squared_var(
  //       num_points_3d);
  //   get(get<conformal_factor_tag>(conformal_factor_squared_var)) =
  //       square(conformal_factor);
  //   const auto d_conformal_factor_squared_var =
  //       partial_derivatives<tmpl::list<conformal_factor_tag>>(
  //           conformal_factor_squared_var, mesh,
  //           coord_map.inv_jacobian(x_logical));
  //   const auto& d_conformal_factor_squared =
  //       get<Tags::deriv<conformal_factor_tag, tmpl::size_t<SpatialDim>,
  //                       Frame::Inertial>>(d_conformal_factor_squared_var);

  //   const auto& d_spatial_metric =
  //       get<Tags::deriv<gr::Tags::SpatialMetric<SpatialDim>,
  //                       tmpl::size_t<SpatialDim>, Frame::Inertial>>(vars);

  //   tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>
  //       expected_d_conformal_spatial_metric{};
  //   for (size_t k = 0; k < SpatialDim; k++) {
  //     for (size_t i = 0; i < SpatialDim; i++) {
  //       for (size_t j = i; j < SpatialDim; j++) {
  //         expected_d_conformal_spatial_metric.get(k, i, j) =
  //             // 2.0 * d_conformal_factor.get(k) *
  //             d_conformal_factor_squared.get(k) * spatial_metric.get(i, j) +
  //             square(conformal_factor) * d_spatial_metric.get(k, i, j);
  //       }
  //     }
  //   }

  // At best passes eith 1e-11 (with 8 grid points)
  // Note that using the expected_ value going forward does not improve
  // the necessary epsilon of 1e-9 for the ricci tensor, and increasing grid
  // points to 10 (as well as using expected_) foes not fix the issue
  // CHECK_ITERABLE_APPROX(expected_d_conformal_spatial_metric,
  // d_conformal_spatial_metric);
  //   Approx approx0 = Approx::custom().epsilon(1e-11).scale(1.0);
  //   CHECK_ITERABLE_CUSTOM_APPROX(expected_d_conformal_spatial_metric,
  //                                d_conformal_spatial_metric, approx0);

  tnsr::ijj<DataVector, SpatialDim, Frame::Inertial> field_d{};
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        // field_d.get(k, i, j) = 0.5 * d_conformal_spatial_metric.get(k, i, j);
        field_d.get(k, i, j) = 0.5 * d_conformal_spatial_metric.get(k, i, j);
      }
    }
  }

  auto field_d_up = gr::deriv_inverse_spatial_metric(
      inverse_conformal_spatial_metric, field_d);
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        field_d_up.get(k, i, j) *= -1.0;
      }
    }
  }
  //   const DataVector used_for_size =
  //       DataVector(num_points_3d,
  //       std::numeric_limits<double>::signaling_NaN());
  //   auto expected_field_d_up =
  //       make_with_value<tnsr::iJJ<DataVector, SpatialDim, Frame::Inertial>>(
  //           used_for_size, 0.0);
  //   for (size_t k = 0; k < SpatialDim; k++) {
  //     for (size_t i = 0; i < SpatialDim; i++) {
  //       for (size_t j = i; j < SpatialDim; j++) {
  //         for (size_t n = 0; n < SpatialDim; n++) {
  //           for (size_t m = 0; m < SpatialDim; m++) {
  //             expected_field_d_up.get(k, i, j) +=
  //                 0.5 * inverse_conformal_spatial_metric.get(i, n) *
  //                 inverse_conformal_spatial_metric.get(m, j) *
  //                 d_conformal_spatial_metric.get(k, n, m);
  //             // expected_d_conformal_spatial_metric.get(k, n, m);
  //           }
  //         }
  //       }
  //     }
  //   }

  // passes
  //   CHECK_ITERABLE_APPROX(expected_field_d_up, field_d_up);

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

  //   using conformal_factor_tag = Ccz4::Tags::ConformalFactor<DataVector>;
  //   Variables<tmpl::list<conformal_factor_tag>> conformal_factor_var(
  //       num_points_3d);
  //   get(get<conformal_factor_tag>(conformal_factor_var)) = conformal_factor;
  //   const auto d_conformal_factor_var =
  //       partial_derivatives<tmpl::list<conformal_factor_tag>>(
  //           conformal_factor_var, mesh, coord_map.inv_jacobian(x_logical));
  //   const auto& d_conformal_factor =
  //       get<Tags::deriv<conformal_factor_tag, tmpl::size_t<SpatialDim>,
  //                       Frame::Inertial>>(d_conformal_factor_var);

  tnsr::i<DataVector, SpatialDim, Frame::Inertial> field_p{};
  for (size_t i = 0; i < SpatialDim; i++) {
    field_p.get(i) =
        -d_det_spatial_metric.get(i) / (6. * get(det_spatial_metric));
    // field_p.get(i) = d_conformal_factor.get(i) / conformal_factor;
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

  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);

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
  // passes with 1e-12 at best after using analytical calculation of
  // d_det_spatial_metric from KerrSchild
  Approx approx1 = Approx::custom().epsilon(1e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_d_conformal_christoffel_second_kind,
                               d_conformal_christoffel_second_kind, approx1);

  const auto christoffel_second_kind = Ccz4::christoffel_second_kind(
      conformal_spatial_metric, inverse_conformal_spatial_metric, field_p,
      conformal_christoffel_second_kind);
  //   const auto& d_spatial_metric =
  //       get<Tags::deriv<gr::Tags::SpatialMetric<SpatialDim>,
  //                       tmpl::size_t<SpatialDim>, Frame::Inertial>>(vars);
  const auto expected_christoffel_second_kind =
      gr::christoffel_second_kind(d_spatial_metric, inverse_spatial_metric);
  // passes with 1e-12 at best with 8 grid points
  // 1e-12 -> 1e-15 after using d_det_spatial_metric for field_p
  Approx approx2 = Approx::custom().epsilon(1e-15).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_christoffel_second_kind,
                               christoffel_second_kind, approx2);

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

  // 1e-9 -> 1e-11 after using analytical computation of
  // d_det_spatial_metric for field_d
  // 1e-11 -> 1e-12 after using analytical computation of
  // d_det_spatial_metric for field_p
  Approx approx = Approx::custom().epsilon(1e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_cpp_gr_ricci_tensor,
                               actual_ricci_tensor, approx);
  // std::cout << grid_size_each_dimension;
  // if (grid_size_each_dimension <= 9) {
  // std::cout << "             | ";
  // } else {
  // std::cout << "            | ";
  // }
  //   std::cout << l2Norm(expected_cpp_gr_ricci_tensor.get(1, 1) -
  //   actual_ricci_tensor.get(1, 1)) << std::endl;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Ricci", "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  const double mass = 2.;
  const std::array<double, 3> spin{{0.3, 0.5, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};
  //   const std::array<double, 3> lower_bound{{8.2, 12.4, 13.2}};
  //   const std::array<double, 3> upper_bound{{8.0, 12.2, 13.0}};

  test_compute_spatial_ricci_tensor(solution, grid_size, lower_bound,
                                    upper_bound);

  //   std::cout << "Num points 1D | L2 norm" << std::endl;
  //   std::cout << "---------------------------------" << std::endl;
  //   test_compute_spatial_ricci_tensor(solution, 2, lower_bound,
  //                                     upper_bound);
  //   test_compute_spatial_ricci_tensor(solution, 3, lower_bound,
  //                                     upper_bound);
  //   test_compute_spatial_ricci_tensor(solution, 4, lower_bound,
  //                                     upper_bound);
  //   test_compute_spatial_ricci_tensor(solution, 5, lower_bound,
  //                                     upper_bound);
  //   test_compute_spatial_ricci_tensor(solution, 6, lower_bound,
  //                                     upper_bound);
  //   test_compute_spatial_ricci_tensor(solution, 7, lower_bound,
  //                                     upper_bound);
  //   test_compute_spatial_ricci_tensor(solution, 8, lower_bound,
  //                                     upper_bound);
  //   test_compute_spatial_ricci_tensor(solution, 9, lower_bound,
  //                                     upper_bound);
  //   test_compute_spatial_ricci_tensor(solution, 10, lower_bound,
  //                                     upper_bound);
  //   test_compute_spatial_ricci_tensor(solution, 11, lower_bound,
  //                                     upper_bound);
  //   test_compute_spatial_ricci_tensor(solution, 12, lower_bound,
  //                                     upper_bound);
}
