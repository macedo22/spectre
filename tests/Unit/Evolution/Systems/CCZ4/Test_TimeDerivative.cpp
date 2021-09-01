// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CCZ4/TimeDerivative.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/CCZ4/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void compute_expected_time_derivative(
    const gsl::not_null<tnsr::ij<DataVector, Dim>*> dt_conf_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> dt_ln_lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_shift,
    const gsl::not_null<Scalar<DataVector>*> dt_ln_phi,
    const gsl::not_null<Scalar<DataVector>*> det_conf_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> trace_A_tilde,
    const tnsr::ii<DataVector, Dim>& conf_spatial_metric,
    const tnsr::I<DataVector, Dim>& shift,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& K_0, const tnsr::i<DataVector, Dim>& A,
    const tnsr::ijk<DataVector, Dim>& D, const tnsr::iJ<DataVector, Dim>& B,
    const tnsr::i<DataVector, Dim>& P, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& g, const Scalar<DataVector>& theta,
    const double c, const tnsr::ij<DataVector, Dim>& A_tilde,
    const double relaxation_time, const double s, const double f,
    const tnsr::I<DataVector, Dim>& b) noexcept {
  // time derivative of the conformal spatial metric
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      (*dt_conf_spatial_metric).get(i, j) =
          -2.0 * lapse.get() *
              (A_tilde.get(i, j) - (1.0 / 3) * conf_spatial_metric.get(i, j) *
                                       (*trace_A_tilde).get()) -
          (1.0 / relaxation_time) * ((*det_conf_spatial_metric).get() - 1.0) *
              conf_spatial_metric.get(i, j);
    }
  }

  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      for (size_t k = 0; k < Dim; k++) {
        (*dt_conf_spatial_metric).get(i, j) +=
            2.0 * shift.get(k) * D.get(k, i, j) +
            conf_spatial_metric.get(k, i) * B.get(j, k) +
            conf_spatial_metric.get(k, j) * B.get(i, k) -
            (2.0 / 3) * conf_spatial_metric.get(i, j) * B.get(k, k);
      }
    }
  }

  // dt_ln_lapse: time derivative of the natural log of the lapse
  (*dt_ln_lapse).get() =
      -lapse.get() * g.get() *
      (trace_extrinsic_curvature.get() - K_0.get() - 2.0 * theta.get() * c);
  for (size_t k = 0; k < Dim; k++) {
    (*dt_ln_lapse).get() += shift.get(k) * A.get(k);
  }

  // dt_shift: time derivative of the shift
  for (size_t i = 0; i < Dim; i++) {
    (*dt_shift).get(i) = s * f * b.get(i);
    for (size_t k = 0; k < Dim; k++) {
      (*dt_shift).get(i) += s * shift.get(k) * B.get(k, i);
    }
  }

  // dt_ln_phi: time derivative of the natural log of the conformal factor
  (*dt_ln_phi).get() =
      (1.0 / 3) * lapse.get() * trace_extrinsic_curvature.get();
  for (size_t k = 0; k < Dim; k++) {
    (*dt_ln_phi).get() += shift.get(k) * P.get(k) - (1.0 / 3) * B.get(k, k);
  }
}

template <size_t Dim, typename Generator>
void test_time_derivative(const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  using ccz4_tags_list =
      tmpl::list<gr::Tags::SpatialMetric<Dim>, gr::Tags::Shift<Dim>,
                 gr::Tags::Lapse<DataVector>, CCZ4::Tags::Phi<DataVector>>;

  const size_t num_grid_points_1d = 3;
  const Mesh<Dim> mesh(num_grid_points_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const DataVector used_for_size(mesh.number_of_grid_points());

  Variables<ccz4_tags_list> evolved_vars(mesh.number_of_grid_points());

  auto& spatial_metric = get<gr::Tags::SpatialMetric<Dim>>(evolved_vars);
  spatial_metric =
      TestHelpers::gr::random_spatial_metric<Dim>(generator, used_for_size);
  Scalar<DataVector> det_spatial_metric(used_for_size);
  tnsr::II<DataVector, Dim> inv_spatial_metric(used_for_size);
  determinant_and_inverse(make_not_null(&det_spatial_metric),
                          make_not_null(&inv_spatial_metric), spatial_metric);
  auto& phi = get<CCZ4::Tags::Phi<DataVector>>(evolved_vars);
  get(phi) = 1.0 / pow(get(det_spatial_metric), 1.0 / 6);
  Scalar<DataVector> phi_squared(used_for_size);
  get(phi_squared) = get(phi) * get(phi);
  tnsr::ii<DataVector, Dim>& conf_spatial_metric =
      get<gr::Tags::SpatialMetric<Dim>>(evolved_vars);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      conf_spatial_metric.get(i, j) =
          get(phi_squared) * spatial_metric.get(i, j);
    }
  }

  auto& shift = get<gr::Tags::Shift<Dim>>(evolved_vars);
  auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
  shift = TestHelpers::gr::random_shift<Dim>(generator, used_for_size);
  lapse = TestHelpers::gr::random_lapse(generator, used_for_size);

  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      if (i == j) {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 1.0);
      } else {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 0.0);
      }
    }
  }

  const auto partial_derivs =
      partial_derivatives<ccz4_tags_list>(evolved_vars, mesh, inv_jac);

  const auto& d_conf_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  tnsr::ijk<DataVector, Dim> D(used_for_size);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t k = 0; k < Dim; k++) {
      for (size_t j = 0; j < Dim; j++) {
        D.get(k, i, j) = 0.5 * d_conf_spatial_metric.get(k, i, j);
      }
    }
  }

  const auto& B = get<
      Tags::deriv<gr::Tags::Shift<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>(
      partial_derivs);

  const auto& A =
      get<Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);

  const auto& P =
      get<Tags::deriv<CCZ4::Tags::Phi<DataVector>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);

  Scalar<DataVector> det_conf_spatial_metric(used_for_size);
  tnsr::II<DataVector, Dim> inv_conf_spatial_metric(used_for_size);
  determinant_and_inverse(make_not_null(&det_conf_spatial_metric),
                          make_not_null(&inv_conf_spatial_metric),
                          conf_spatial_metric);

  const auto extrinsic_curvature =
      make_with_random_values<tnsr::ii<DataVector, Dim>>(
          generator, make_not_null(&distribution), used_for_size);
  auto trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      trace_extrinsic_curvature.get() +=
          extrinsic_curvature.get(i, j) * inv_spatial_metric.get(i, j);
    }
  }

  tnsr::ij<DataVector, Dim> A_tilde(used_for_size);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      A_tilde.get(i, j) =
          get(phi_squared) * (extrinsic_curvature.get(i, j) -
                              (1.0 / 3) * trace_extrinsic_curvature.get() *
                                  spatial_metric.get(i, j));
    }
  }

  auto trace_A_tilde = make_with_value<Scalar<DataVector>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      trace_A_tilde.get() +=
          inv_conf_spatial_metric.get(i, j) * A_tilde.get(i, j);
    }
  }

  const auto K_0 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const auto g = make_with_value<Scalar<DataVector>>(used_for_size, 1.);
  const auto theta = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const double c = 1.0;
  const double relaxation_time = 1.2;
  const double s = 1.3;
  const double f = 2.1;
  const auto b = make_with_random_values<tnsr::I<DataVector, Dim>>(
      generator, make_not_null(&distribution), used_for_size);

  tnsr::ij<DataVector, Dim> dt_conf_spatial_metric(used_for_size);
  Scalar<DataVector> dt_ln_lapse(used_for_size);
  tnsr::I<DataVector, Dim> dt_shift(used_for_size);
  Scalar<DataVector> dt_ln_phi(used_for_size);

  CCZ4::TimeDerivative<Dim, DataVector>::apply(
      make_not_null(&dt_conf_spatial_metric), make_not_null(&dt_ln_lapse),
      make_not_null(&dt_shift), make_not_null(&dt_ln_phi),
      make_not_null(&det_conf_spatial_metric), make_not_null(&trace_A_tilde),
      conf_spatial_metric, shift, trace_extrinsic_curvature, K_0, A, D, B, P,
      lapse, g, theta, c, A_tilde, relaxation_time, s, f, b);

  tnsr::ij<DataVector, Dim> expected_dt_conf_spatial_metric(used_for_size);
  Scalar<DataVector> expected_dt_ln_lapse(used_for_size);
  tnsr::I<DataVector, Dim> expected_dt_shift(used_for_size);
  Scalar<DataVector> expected_dt_ln_phi(used_for_size);

  compute_expected_time_derivative(
      make_not_null(&expected_dt_conf_spatial_metric),
      make_not_null(&expected_dt_ln_lapse), make_not_null(&expected_dt_shift),
      make_not_null(&expected_dt_ln_phi),
      make_not_null(&det_conf_spatial_metric), make_not_null(&trace_A_tilde),
      conf_spatial_metric, shift, trace_extrinsic_curvature, K_0, A, D, B, P,
      lapse, g, theta, c, A_tilde, relaxation_time, s, f, b);

  CHECK_ITERABLE_APPROX(dt_conf_spatial_metric,
                        expected_dt_conf_spatial_metric);
  CHECK_ITERABLE_APPROX(dt_ln_lapse, expected_dt_ln_lapse);
  CHECK_ITERABLE_APPROX(dt_shift, expected_dt_shift);
  CHECK_ITERABLE_APPROX(dt_ln_phi, expected_dt_ln_phi);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.DuDt",
                  "[Unit][GeneralizedHarmonic]") {
  MAKE_GENERATOR(generator);

  test_time_derivative<1>(make_not_null(&generator));
  test_time_derivative<2>(make_not_null(&generator));
  test_time_derivative<3>(make_not_null(&generator));
}
