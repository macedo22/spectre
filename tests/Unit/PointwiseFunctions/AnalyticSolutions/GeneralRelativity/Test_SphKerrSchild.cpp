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
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include <cmath>
#include <iostream>
#include <typeinfo>

// IWYU pragma: no_forward_declare Tags::deriv

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x)[0] = 1.1001;
  get<0>(x)[1] = 1.1;
  get<0>(x)[2] = 1.1;
  get<1>(x)[0] = 3.2;
  get<1>(x)[1] = 3.2001;
  get<1>(x)[2] = 3.2;
  get<2>(x)[0] = 5.3;
  get<2>(x)[1] = 5.3;
  get<2>(x)[2] = 5.3001;
  return x;
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.SphKerrSchild",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/");

  // Parameters for SphKerrSchild solution
  const DataVector used_for_size(3);
  const size_t used_for_sizet = used_for_size.size();
  const double mass = 1.01;
  const std::array<double, 3> spin{{0.2, 0.3, 0.4}};
  const std::array<double, 3> center{{0.1, 1.2, 2.3}};
  const auto x = spatial_coords<Frame::Inertial>(used_for_size);
  // const double null_vector_0 = -1.0;
  // const double t = 1.0;

  // Set up the solution, computer object, and cache object
  gr::Solutions::SphKerrSchild solution(mass, spin, center);
  gr::Solutions::SphKerrSchild::IntermediateComputer sks_computer(solution, x);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache(used_for_sizet);

  // test functions

  // x_sph_minus_center test
  auto x_sph_minus_center = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&x_sph_minus_center), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::x_sph_minus_center<
                   DataVector, Frame::Inertial>{});

  // r_squared test
  Scalar<DataVector> r_squared(3_st, 0.);
  sks_computer(
      make_not_null(&r_squared), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::r_squared<DataVector>{});

  // r test
  Scalar<DataVector> r(3_st, 0.);
  sks_computer(make_not_null(&r), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::r<DataVector>{});

  // rho test
  Scalar<DataVector> rho(3_st, 0.);
  sks_computer(make_not_null(&rho), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::rho<DataVector>{});

  // matrix_F test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_F{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_F), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_F<DataVector,
                                                            Frame::Inertial>{});

  // matrix_P test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_P{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_P), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_P<DataVector,
                                                            Frame::Inertial>{});

  //   // jacobian test
  //   tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian{1_st, 0.};
  //   sks_computer(
  //       make_not_null(&jacobian), make_not_null(&cache),
  //       gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
  //                                                   Frame::Inertial>{});

  // matrix_D test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_D{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_D), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_D<DataVector,
                                                            Frame::Inertial>{});

  // matrix_C test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_C{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_C), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_C<DataVector,
                                                            Frame::Inertial>{});

  //   // deriv_jacobian test
  //   tnsr::ijK<DataVector, 3, Frame::Inertial> deriv_jacobian{1_st, 0.};
  //   sks_computer(make_not_null(&deriv_jacobian), make_not_null(&cache),
  //                gr::Solutions::SphKerrSchild::internal_tags::deriv_jacobian<
  //                    DataVector, Frame::Inertial>{});

  // Setup grid
  const size_t num_points_1d = 8;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{num_points_1d, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},

              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });
  const size_t num_points_3d = num_points_1d * num_points_1d * num_points_1d;
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x_prime = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate analytic solution
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian{1_st, 0.};
  sks_computer(
      make_not_null(&jacobian), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  // Compute actual analytical derivative of the determinant
  tnsr::ijK<DataVector, 3, Frame::Inertial> deriv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&deriv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_jacobian<
                   DataVector, Frame::Inertial>{});

  // Compute expected numerical derivative of the jaccobian
  using jacobian_tag =
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>;
  Variables<tmpl::list<jacobian_tag>> jacobian_var(num_points_3d);
  get<jacobian_tag>(jacobian_var) = jacobian;
  const auto expected_deriv_jacobian_var =
      partial_derivatives<tmpl::list<jacobian_tag>>(
          jacobian_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& expected_deriv_jacobian =
      get<Tags::deriv<jacobian_tag, tmpl::size_t<SpatialDim>, Frame::Inertial>>(
          expected_deriv_jacobian_var);

  Approx custom_approx = Approx::custom().epsilon(1e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(deriv_jacobian, expected_deriv_jacobian,
                               custom_approx);

  //   // matrix_Q test
  //   tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_Q{1_st, 0.};
  //   sks_computer(
  //       make_not_null(&matrix_Q), make_not_null(&cache),
  //       gr::Solutions::SphKerrSchild::internal_tags::matrix_Q<DataVector,
  //                                                    Frame::Inertial>{});

  //   // matrix_G1 test
  //   tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_G1{1_st, 0.};
  //   sks_computer(make_not_null(&matrix_G1), make_not_null(&cache),
  //                gr::Solutions::SphKerrSchild::internal_tags::matrix_G1<
  //                    DataVector, Frame::Inertial>{});

  //   // a_dot_x test
  //   Scalar<DataVector> a_dot_x(3_st, 0.);
  //   sks_computer(
  //       make_not_null(&a_dot_x), make_not_null(&cache),
  //       gr::Solutions::SphKerrSchild::internal_tags::a_dot_x<DataVector>{});

  //   // matrix_G2 test
  //   tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_G2{1_st, 0.};
  //   sks_computer(make_not_null(&matrix_G2), make_not_null(&cache),
  //                gr::Solutions::SphKerrSchild::internal_tags::matrix_G2<
  //                    DataVector, Frame::Inertial>{});

  //   // G1_dot_x test
  //   tnsr::I<DataVector, 3, Frame::Inertial> G1_dot_x{3_st, 0.};
  //   sks_computer(
  //       make_not_null(&G1_dot_x), make_not_null(&cache),
  //       gr::Solutions::SphKerrSchild::internal_tags::G1_dot_x<DataVector,
  //                                                      Frame::Inertial>{});

  //   // G2_dot_x test
  //   tnsr::i<DataVector, 3, Frame::Inertial> G2_dot_x{3_st, 0.};
  //   sks_computer(
  //       make_not_null(&G2_dot_x), make_not_null(&cache),
  //       gr::Solutions::SphKerrSchild::internal_tags::G2_dot_x<DataVector,
  //                                                      Frame::Inertial>{});

  //   // inv_jacobian test
  //   tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian{1_st, 0.};
  //   sks_computer(make_not_null(&inv_jacobian), make_not_null(&cache),
  //                gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
  //                    DataVector, Frame::Inertial>{});

  //   // matrix_E1 test
  //   tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_E1{1_st, 0.};
  //   sks_computer(make_not_null(&matrix_E1), make_not_null(&cache),
  //                gr::Solutions::SphKerrSchild::internal_tags::matrix_E1<
  //                    DataVector, Frame::Inertial>{});

  //   // matrix_E2 test
  //   tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_E2{1_st, 0.};
  //   sks_computer(make_not_null(&matrix_E2), make_not_null(&cache),
  //                gr::Solutions::SphKerrSchild::internal_tags::matrix_E2<
  //                    DataVector, Frame::Inertial>{});

  //   // deriv_inv_jacobian test
  //   tnsr::ijK<DataVector, 3, Frame::Inertial> deriv_inv_jacobian{1_st, 0.};
  //   sks_computer(make_not_null(&deriv_inv_jacobian), make_not_null(&cache),
  //           gr::Solutions::SphKerrSchild::internal_tags::deriv_inv_jacobian<
  //                    DataVector, Frame::Inertial>{});

  // David's code to test the General_Finite_Difference.py file to return the
  // jacobian

  //   const tnsr::I<DataVector, 3, Frame::Inertial>& pert_coords_wrong_shape =
  //       cache.get_var(gr::Solutions::SphKerrSchild::internal_tags::
  // x_kerr_schild<
  //                     DataVector, Frame::Inertial>{});

  //   tnsr::Ij<DataVector, 3, Frame::Inertial> pert_coords_right_shape{1_st,
  //   0.}; for (size_t i = 0; i < 3; ++i) {
  //     for (size_t j = 0; j < 3; ++j) {
  //       pert_coords_right_shape.get(i, j) = pert_coords_wrong_shape[j][i];
  //     }
  //   }

  //   auto input_coords =
  //       make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1_st, 0.0);
  //   input_coords[0] = 0.9960134139755227;
  //   input_coords[1] = 1.999275166177368;
  //   input_coords[2] = 3.002536918379212;

  //   auto pertubation =
  //       make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1_st, 0.0001);

  //   const auto finite_diff_jacobian =
  //       pypp::call<tnsr::Ij<DataVector, 3, Frame::Inertial>>(
  //           "General_Finite_Difference", "check_finite_difference",
  //           input_coords, pert_coords_right_shape, pertubation);

  //   std::cout << "JACOBIAN???!!!"
  //             << "\n"
  //             << finite_diff_jacobian << "\n";
}
