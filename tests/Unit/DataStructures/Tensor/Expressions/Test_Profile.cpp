// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <random>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Executables/Benchmark/BenchmarkedImpls.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename... Ts>
void copy_tensor(const Tensor<Ts...>& tensor_source,
                 gsl::not_null<Tensor<Ts...>*> tensor_destination) noexcept {
  auto tensor_source_it = tensor_source.begin();
  auto tensor_destination_it = tensor_destination->begin();
  for (; tensor_source_it != tensor_source.end();
       tensor_source_it++, tensor_destination_it++) {
    *tensor_destination_it = *tensor_source_it;
  }
}
}  // namespace

// Make sure TE impl matches manual impl
template <size_t Dim, typename DataType, typename Generator>
void test_benchmarked_implementations_core(
    const gsl::not_null<Generator*> generator) noexcept {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using dt_pi_type = typename BenchmarkImpl::dt_pi_type;
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using temp_gamma1_type = typename BenchmarkImpl::temp_gamma1_type;
  using temp_gamma2_type = typename BenchmarkImpl::temp_gamma2_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using pi_two_normals_type = typename BenchmarkImpl::pi_two_normals_type;
  using normal_dot_gauge_constraint_type =
      typename BenchmarkImpl::normal_dot_gauge_constraint_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using pi_one_normal_type = typename BenchmarkImpl::pi_one_normal_type;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using pi_2_up_type = typename BenchmarkImpl::pi_2_up_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using phi_3_up_type = typename BenchmarkImpl::phi_3_up_type;
  using christoffel_first_kind_3_up_type =
      typename BenchmarkImpl::christoffel_first_kind_3_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using spatial_metric_type = typename BenchmarkImpl::spatial_metric_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using det_spatial_metric_type =
      typename BenchmarkImpl::det_spatial_metric_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using christoffel_second_kind_type =
      typename BenchmarkImpl::christoffel_second_kind_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using normal_spacetime_vector_type =
      typename BenchmarkImpl::normal_spacetime_vector_type;
  using normal_spacetime_one_form_type =
      typename BenchmarkImpl::normal_spacetime_one_form_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;
  using d_spacetime_metric_type =
      typename BenchmarkImpl::d_spacetime_metric_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using spacetime_metric_type = typename BenchmarkImpl::spacetime_metric_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using phi_type = typename BenchmarkImpl::phi_type;
  using gamma0_type = typename BenchmarkImpl::gamma0_type;
  using gamma1_type = typename BenchmarkImpl::gamma1_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using spacetime_deriv_gauge_function_type =
      typename BenchmarkImpl::spacetime_deriv_gauge_function_type;
  // types not in Spectre implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using phi_one_normal_spatial_type =
      typename BenchmarkImpl::phi_one_normal_spatial_type;

  // Set up input constant tensors in equations
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  using gh_tags_list = tmpl::list<gr::Tags::SpacetimeMetric<Dim>,
                                  GeneralizedHarmonic::Tags::Pi<Dim>,
                                  GeneralizedHarmonic::Tags::Phi<Dim>>;

  const size_t num_grid_points_1d = 3;
  const Mesh<Dim> mesh(num_grid_points_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const DataVector used_for_size(mesh.number_of_grid_points());

  Variables<gh_tags_list> evolved_vars(mesh.number_of_grid_points());
  fill_with_random_values(make_not_null(&evolved_vars), generator,
                          make_not_null(&distribution));
  // In order to satisfy the physical requirements on the spacetime metric we
  // compute it from the helper functions that generate a physical lapse, shift,
  // and spatial metric.
  gr::spacetime_metric(
      make_not_null(&get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars)),
      TestHelpers::gr::random_lapse(generator, used_for_size),
      TestHelpers::gr::random_shift<Dim>(generator, used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(generator, used_for_size));

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
      partial_derivatives<gh_tags_list>(evolved_vars, mesh, inv_jac);

  const spacetime_metric_type& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars);
  const phi_type& phi = get<GeneralizedHarmonic::Tags::Phi<Dim>>(evolved_vars);
  const pi_type& pi = get<GeneralizedHarmonic::Tags::Pi<Dim>>(evolved_vars);
  const d_spacetime_metric_type& d_spacetime_metric =
      get<Tags::deriv<gr::Tags::SpacetimeMetric<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const d_phi_type& d_phi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Phi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const d_pi_type& d_pi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Pi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  ;

  const gamma0_type gamma0 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const gamma1_type gamma1 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const gamma2_type gamma2 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const gauge_function_type gauge_function =
      make_with_random_values<tnsr::a<DataVector, Dim>>(
          generator, make_not_null(&distribution), used_for_size);
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<tnsr::ab<DataVector, Dim>>(
          generator, make_not_null(&distribution), used_for_size);

  const size_t num_grid_points = mesh.number_of_grid_points();

  // Initialize non-const tensors to be filled by manual implementation
  dt_spacetime_metric_type dt_spacetime_metric_manual(num_grid_points);
  dt_pi_type dt_pi_manual(num_grid_points);
  dt_phi_type dt_phi_manual(num_grid_points);
  temp_gamma1_type temp_gamma1_manual(num_grid_points);
  temp_gamma2_type temp_gamma2_manual(num_grid_points);
  gamma1gamma2_type gamma1gamma2_manual(num_grid_points);
  pi_two_normals_type pi_two_normals_manual(num_grid_points);
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint_manual(
      num_grid_points);
  gamma1_plus_1_type gamma1_plus_1_manual(num_grid_points);
  pi_one_normal_type pi_one_normal_manual(num_grid_points);
  gauge_constraint_type gauge_constraint_manual(num_grid_points);
  phi_two_normals_type phi_two_normals_manual(num_grid_points);
  shift_dot_three_index_constraint_type shift_dot_three_index_constraint_manual(
      num_grid_points);
  phi_one_normal_type phi_one_normal_manual(num_grid_points);
  pi_2_up_type pi_2_up_manual(num_grid_points);
  three_index_constraint_type three_index_constraint_manual(num_grid_points);
  phi_1_up_type phi_1_up_manual(num_grid_points);
  phi_3_up_type phi_3_up_manual(num_grid_points);
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up_manual(
      num_grid_points);
  lapse_type lapse_manual(num_grid_points);
  shift_type shift_manual(num_grid_points);
  spatial_metric_type spatial_metric_manual(num_grid_points);
  inverse_spatial_metric_type inverse_spatial_metric_manual(num_grid_points);
  det_spatial_metric_type det_spatial_metric_manual(num_grid_points);
  inverse_spacetime_metric_type inverse_spacetime_metric_manual(
      num_grid_points);
  christoffel_first_kind_type christoffel_first_kind_manual(num_grid_points);
  christoffel_second_kind_type christoffel_second_kind_manual(num_grid_points);
  trace_christoffel_type trace_christoffel_manual(num_grid_points);
  normal_spacetime_vector_type normal_spacetime_vector_manual(num_grid_points);
  normal_spacetime_one_form_type normal_spacetime_one_form_manual(
      num_grid_points);
  da_spacetime_metric_type da_spacetime_metric_manual(num_grid_points);
  // TEs can't iterate over only spatial indices of a spacetime index yet, so
  // where this is needed for the dt_pi and dt_phi calculations, these tensors
  // will be used, which hold only the spatial components to enable writing the
  // equations as closely as possible to how they appear in the manual loops
  pi_one_normal_spatial_type pi_one_normal_spatial_manual(num_grid_points);
  phi_one_normal_spatial_type phi_one_normal_spatial_manual(num_grid_points);

  // Initialize non-const tensors to be filled by TensorExpression
  // implementation
  dt_spacetime_metric_type dt_spacetime_metric_te_filled(num_grid_points);
  dt_pi_type dt_pi_te_filled(num_grid_points);
  dt_phi_type dt_phi_te_filled(num_grid_points);
  temp_gamma1_type temp_gamma1_te_filled(num_grid_points);
  temp_gamma2_type temp_gamma2_te_filled(num_grid_points);
  gamma1gamma2_type gamma1gamma2_te_filled(num_grid_points);
  pi_two_normals_type pi_two_normals_te_filled(num_grid_points);
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint_te_filled(
      num_grid_points);
  gamma1_plus_1_type gamma1_plus_1_te_filled(num_grid_points);
  pi_one_normal_type pi_one_normal_te_filled(num_grid_points);
  gauge_constraint_type gauge_constraint_te_filled(num_grid_points);
  phi_two_normals_type phi_two_normals_te_filled(num_grid_points);
  shift_dot_three_index_constraint_type
      shift_dot_three_index_constraint_te_filled(num_grid_points);
  phi_one_normal_type phi_one_normal_te_filled(num_grid_points);
  pi_2_up_type pi_2_up_te_filled(num_grid_points);
  three_index_constraint_type three_index_constraint_te_filled(num_grid_points);
  phi_1_up_type phi_1_up_te_filled(num_grid_points);
  phi_3_up_type phi_3_up_te_filled(num_grid_points);
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up_te_filled(
      num_grid_points);
  lapse_type lapse_te_filled(num_grid_points);
  shift_type shift_te_filled(num_grid_points);
  spatial_metric_type spatial_metric_te_filled(num_grid_points);
  inverse_spatial_metric_type inverse_spatial_metric_te_filled(num_grid_points);
  det_spatial_metric_type det_spatial_metric_te_filled(num_grid_points);
  inverse_spacetime_metric_type inverse_spacetime_metric_te_filled(
      num_grid_points);
  christoffel_first_kind_type christoffel_first_kind_te_filled(num_grid_points);
  christoffel_second_kind_type christoffel_second_kind_te_filled(
      num_grid_points);
  trace_christoffel_type trace_christoffel_te_filled(num_grid_points);
  normal_spacetime_vector_type normal_spacetime_vector_te_filled(
      num_grid_points);
  normal_spacetime_one_form_type normal_spacetime_one_form_te_filled(
      num_grid_points);
  da_spacetime_metric_type da_spacetime_metric_te_filled(num_grid_points);
  // TEs can't iterate over only spatial indices of a spacetime index yet, so
  // where this is needed for the dt_pi and dt_phi calculations, these tensors
  // will be used, which hold only the spatial components to enable writing the
  // equations as closely as possible to how they appear in the manual loops
  pi_one_normal_spatial_type pi_one_normal_spatial_te_filled(num_grid_points);
  phi_one_normal_spatial_type phi_one_normal_spatial_te_filled(num_grid_points);

  // Compute manual result
  BenchmarkImpl::manual_impl_lhs_as_arg(
      make_not_null(&dt_spacetime_metric_manual), make_not_null(&dt_pi_manual),
      make_not_null(&dt_phi_manual), make_not_null(&temp_gamma1_manual),
      make_not_null(&temp_gamma2_manual), make_not_null(&gamma1gamma2_manual),
      make_not_null(&pi_two_normals_manual),
      make_not_null(&normal_dot_gauge_constraint_manual),
      make_not_null(&gamma1_plus_1_manual),
      make_not_null(&pi_one_normal_manual),
      make_not_null(&gauge_constraint_manual),
      make_not_null(&phi_two_normals_manual),
      make_not_null(&shift_dot_three_index_constraint_manual),
      make_not_null(&phi_one_normal_manual), make_not_null(&pi_2_up_manual),
      make_not_null(&three_index_constraint_manual),
      make_not_null(&phi_1_up_manual), make_not_null(&phi_3_up_manual),
      make_not_null(&christoffel_first_kind_3_up_manual),
      make_not_null(&lapse_manual), make_not_null(&shift_manual),
      make_not_null(&spatial_metric_manual),
      make_not_null(&inverse_spatial_metric_manual),
      make_not_null(&det_spatial_metric_manual),
      make_not_null(&inverse_spacetime_metric_manual),
      make_not_null(&christoffel_first_kind_manual),
      make_not_null(&christoffel_second_kind_manual),
      make_not_null(&trace_christoffel_manual),
      make_not_null(&normal_spacetime_vector_manual),
      make_not_null(&normal_spacetime_one_form_manual),
      make_not_null(&da_spacetime_metric_manual), d_spacetime_metric, d_pi,
      d_phi, spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
      spacetime_deriv_gauge_function,
      make_not_null(&pi_one_normal_spatial_manual),
      make_not_null(&phi_one_normal_spatial_manual));

  // Compute TensorExpression result
  BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
      make_not_null(&dt_spacetime_metric_te_filled),
      make_not_null(&dt_pi_te_filled), make_not_null(&dt_phi_te_filled),
      make_not_null(&temp_gamma1_te_filled),
      make_not_null(&temp_gamma2_te_filled),
      make_not_null(&gamma1gamma2_te_filled),
      make_not_null(&pi_two_normals_te_filled),
      make_not_null(&normal_dot_gauge_constraint_te_filled),
      make_not_null(&gamma1_plus_1_te_filled),
      make_not_null(&pi_one_normal_te_filled),
      make_not_null(&gauge_constraint_te_filled),
      make_not_null(&phi_two_normals_te_filled),
      make_not_null(&shift_dot_three_index_constraint_te_filled),
      make_not_null(&phi_one_normal_te_filled),
      make_not_null(&pi_2_up_te_filled),
      make_not_null(&three_index_constraint_te_filled),
      make_not_null(&phi_1_up_te_filled), make_not_null(&phi_3_up_te_filled),
      make_not_null(&christoffel_first_kind_3_up_te_filled),
      make_not_null(&lapse_te_filled), make_not_null(&shift_te_filled),
      make_not_null(&spatial_metric_te_filled),
      make_not_null(&inverse_spatial_metric_te_filled),
      make_not_null(&det_spatial_metric_te_filled),
      make_not_null(&inverse_spacetime_metric_te_filled),
      make_not_null(&christoffel_first_kind_te_filled),
      make_not_null(&christoffel_second_kind_te_filled),
      make_not_null(&trace_christoffel_te_filled),
      make_not_null(&normal_spacetime_vector_te_filled),
      make_not_null(&normal_spacetime_one_form_te_filled),
      make_not_null(&da_spacetime_metric_te_filled), d_spacetime_metric, d_pi,
      d_phi, spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
      spacetime_deriv_gauge_function,
      make_not_null(&pi_one_normal_spatial_te_filled),
      make_not_null(&phi_one_normal_spatial_te_filled));

  // CHECK christoffel_first_kind (abb)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(christoffel_first_kind_manual.get(a, b, c),
                              christoffel_first_kind_te_filled.get(a, b, c));
      }
    }
  }

  // CHECK christoffel_second_kind (Abb)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(christoffel_second_kind_manual.get(a, b, c),
                              christoffel_second_kind_te_filled.get(a, b, c));
      }
    }
  }

  // CHECK trace_christoffel (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(trace_christoffel_manual.get(a),
                          trace_christoffel_te_filled.get(a));
  }

  // CHECK gamma1gamma2 (scalar)
  CHECK_ITERABLE_APPROX(gamma1gamma2_manual.get(),
                        gamma1gamma2_te_filled.get());

  // CHECK phi_1_up (Iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_1_up_manual.get(i, a, b),
                              phi_1_up_te_filled.get(i, a, b));
      }
    }
  }

  // CHECK phi_3_up (iaB)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(phi_3_up_manual.get(i, a, b),
                              phi_3_up_te_filled.get(i, a, b));
      }
    }
  }

  // CHECK pi_2_up (aB)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(pi_2_up_manual.get(a, b),
                            pi_2_up_te_filled.get(a, b));
    }
  }

  // CHECK christoffel_first_kind_3_up (abC)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        CHECK_ITERABLE_APPROX(
            christoffel_first_kind_3_up_manual.get(a, b, c),
            christoffel_first_kind_3_up_te_filled.get(a, b, c));
      }
    }
  }

  // CHECK pi_one_normal (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(pi_one_normal_manual.get(a),
                          pi_one_normal_te_filled.get(a));
  }

  // CHECK pi_one_normal_spatial (i)
  for (size_t i = 0; i < Dim; i++) {
    CHECK_ITERABLE_APPROX(pi_one_normal_spatial_manual.get(i),
                          pi_one_normal_spatial_te_filled.get(i));
  }

  // CHECK pi_two_normals (scalar)
  CHECK_ITERABLE_APPROX(pi_two_normals_manual.get(),
                        pi_two_normals_te_filled.get());

  // CHECK phi_one_normal (ia)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      CHECK_ITERABLE_APPROX(phi_one_normal_manual.get(i, a),
                            phi_one_normal_te_filled.get(i, a));
    }
  }

  // CHECK phi_one_normal_spatial (ij)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      CHECK_ITERABLE_APPROX(phi_one_normal_spatial_manual.get(i, j),
                            phi_one_normal_spatial_te_filled.get(i, j));
    }
  }

  // CHECK phi_two_normals (i)
  for (size_t i = 0; i < Dim; i++) {
    CHECK_ITERABLE_APPROX(phi_two_normals_manual.get(i),
                          phi_two_normals_te_filled.get(i));
  }

  // CHECK three_index_constraint (iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(three_index_constraint_manual.get(i, a, b),
                              three_index_constraint_te_filled.get(i, a, b));
      }
    }
  }

  // CHECK gauge_constraint (a)
  for (size_t a = 0; a < Dim + 1; a++) {
    CHECK_ITERABLE_APPROX(gauge_constraint_manual.get(a),
                          gauge_constraint_te_filled.get(a));
  }

  // CHECK normal_dot_gauge_constraint (scalar)
  CHECK_ITERABLE_APPROX(normal_dot_gauge_constraint_manual.get(),
                        normal_dot_gauge_constraint_te_filled.get());

  // CHECK gamma1_plus_1 (scalar)
  CHECK_ITERABLE_APPROX(gamma1_plus_1_manual.get(),
                        gamma1_plus_1_te_filled.get());

  // CHECK shift_dot_three_index_constraint (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(
          shift_dot_three_index_constraint_manual.get(a, b),
          shift_dot_three_index_constraint_te_filled.get(a, b));
    }
  }

  // CHECK dt_spacetime_metric (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual.get(a, b),
                            dt_spacetime_metric_te_filled.get(a, b));
    }
  }

  // CHECK dt_pi (aa)
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      CHECK_ITERABLE_APPROX(dt_pi_manual.get(a, b), dt_pi_te_filled.get(a, b));
    }
  }

  // CHECK dt_phi (iaa)
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(dt_phi_manual.get(i, a, b),
                              dt_phi_te_filled.get(i, a, b));
      }
    }
  }

  // For DataVectors, check TE implementation with TempTensors
  if constexpr (not std::is_same_v<DataType, double>) {
    TempBuffer<tmpl::list<
        ::Tags::TempTensor<0, dt_spacetime_metric_type>,
        ::Tags::TempTensor<1, dt_pi_type>, ::Tags::TempTensor<2, dt_phi_type>,
        ::Tags::TempTensor<3, temp_gamma1_type>,
        ::Tags::TempTensor<4, temp_gamma2_type>,
        ::Tags::TempTensor<5, gamma1gamma2_type>,
        ::Tags::TempTensor<6, pi_two_normals_type>,
        ::Tags::TempTensor<7, normal_dot_gauge_constraint_type>,
        ::Tags::TempTensor<8, gamma1_plus_1_type>,
        ::Tags::TempTensor<9, pi_one_normal_type>,
        ::Tags::TempTensor<10, gauge_constraint_type>,
        ::Tags::TempTensor<11, phi_two_normals_type>,
        ::Tags::TempTensor<12, shift_dot_three_index_constraint_type>,
        ::Tags::TempTensor<13, phi_one_normal_type>,
        ::Tags::TempTensor<14, pi_2_up_type>,
        ::Tags::TempTensor<15, three_index_constraint_type>,
        ::Tags::TempTensor<16, phi_1_up_type>,
        ::Tags::TempTensor<17, phi_3_up_type>,
        ::Tags::TempTensor<18, christoffel_first_kind_3_up_type>,
        ::Tags::TempTensor<19, lapse_type>, ::Tags::TempTensor<20, shift_type>,
        ::Tags::TempTensor<21, spatial_metric_type>,
        ::Tags::TempTensor<22, inverse_spatial_metric_type>,
        ::Tags::TempTensor<23, det_spatial_metric_type>,
        ::Tags::TempTensor<24, inverse_spacetime_metric_type>,
        ::Tags::TempTensor<25, christoffel_first_kind_type>,
        ::Tags::TempTensor<26, christoffel_second_kind_type>,
        ::Tags::TempTensor<27, trace_christoffel_type>,
        ::Tags::TempTensor<28, normal_spacetime_vector_type>,
        ::Tags::TempTensor<29, normal_spacetime_one_form_type>,
        ::Tags::TempTensor<30, da_spacetime_metric_type>,
        ::Tags::TempTensor<31, d_spacetime_metric_type>,
        ::Tags::TempTensor<32, d_pi_type>, ::Tags::TempTensor<33, d_phi_type>,
        ::Tags::TempTensor<34, spacetime_metric_type>,
        ::Tags::TempTensor<35, pi_type>, ::Tags::TempTensor<36, phi_type>,
        ::Tags::TempTensor<37, gamma0_type>,
        ::Tags::TempTensor<38, gamma1_type>,
        ::Tags::TempTensor<39, gamma2_type>,
        ::Tags::TempTensor<40, gauge_function_type>,
        ::Tags::TempTensor<41, spacetime_deriv_gauge_function_type>,
        ::Tags::TempTensor<42, pi_one_normal_spatial_type>,
        ::Tags::TempTensor<43, phi_one_normal_spatial_type>>>
        vars{num_grid_points};

    // dt_spacetime_metric
    dt_spacetime_metric_type& dt_spacetime_metric_te_temp =
        get<::Tags::TempTensor<0, dt_spacetime_metric_type>>(vars);

    // dt_pi
    dt_pi_type& dt_pi_te_temp = get<::Tags::TempTensor<1, dt_pi_type>>(vars);

    // dt_phi
    dt_phi_type& dt_phi_te_temp = get<::Tags::TempTensor<2, dt_phi_type>>(vars);

    // temp_gamma1
    temp_gamma1_type& temp_gamma1_te_temp =
        get<::Tags::TempTensor<3, temp_gamma1_type>>(vars);

    // temp_gamma2
    temp_gamma2_type& temp_gamma2_te_temp =
        get<::Tags::TempTensor<4, temp_gamma2_type>>(vars);

    // gamma1gamma2
    gamma1gamma2_type& gamma1gamma2_te_temp =
        get<::Tags::TempTensor<5, gamma1gamma2_type>>(vars);

    // pi_two_normals
    pi_two_normals_type& pi_two_normals_te_temp =
        get<::Tags::TempTensor<6, pi_two_normals_type>>(vars);

    // normal_dot_gauge_constraint
    normal_dot_gauge_constraint_type& normal_dot_gauge_constraint_te_temp =
        get<::Tags::TempTensor<7, normal_dot_gauge_constraint_type>>(vars);

    // gamma1_plus_1
    gamma1_plus_1_type& gamma1_plus_1_te_temp =
        get<::Tags::TempTensor<8, gamma1_plus_1_type>>(vars);

    // pi_one_normal
    pi_one_normal_type& pi_one_normal_te_temp =
        get<::Tags::TempTensor<9, pi_one_normal_type>>(vars);

    // gauge_constraint
    gauge_constraint_type& gauge_constraint_te_temp =
        get<::Tags::TempTensor<10, gauge_constraint_type>>(vars);

    // phi_two_normals
    phi_two_normals_type& phi_two_normals_te_temp =
        get<::Tags::TempTensor<11, phi_two_normals_type>>(vars);

    // shift_dot_three_index_constraint
    shift_dot_three_index_constraint_type&
        shift_dot_three_index_constraint_te_temp =
            get<::Tags::TempTensor<12, shift_dot_three_index_constraint_type>>(
                vars);

    // phi_one_normal
    phi_one_normal_type& phi_one_normal_te_temp =
        get<::Tags::TempTensor<13, phi_one_normal_type>>(vars);

    // pi_2_up
    pi_2_up_type& pi_2_up_te_temp =
        get<::Tags::TempTensor<14, pi_2_up_type>>(vars);

    // three_index_constraint
    three_index_constraint_type& three_index_constraint_te_temp =
        get<::Tags::TempTensor<15, three_index_constraint_type>>(vars);

    // phi_1_up
    phi_1_up_type& phi_1_up_te_temp =
        get<::Tags::TempTensor<16, phi_1_up_type>>(vars);

    // phi_3_up
    phi_3_up_type& phi_3_up_te_temp =
        get<::Tags::TempTensor<17, phi_3_up_type>>(vars);

    // christoffel_first_kind_3_up
    christoffel_first_kind_3_up_type& christoffel_first_kind_3_up_te_temp =
        get<::Tags::TempTensor<18, christoffel_first_kind_3_up_type>>(vars);

    // lapse
    lapse_type& lapse_te_temp = get<::Tags::TempTensor<19, lapse_type>>(vars);

    // shift
    shift_type& shift_te_temp = get<::Tags::TempTensor<20, shift_type>>(vars);

    // spatial_metric
    spatial_metric_type& spatial_metric_te_temp =
        get<::Tags::TempTensor<21, spatial_metric_type>>(vars);

    // inverse_spatial_metric
    inverse_spatial_metric_type& inverse_spatial_metric_te_temp =
        get<::Tags::TempTensor<22, inverse_spatial_metric_type>>(vars);

    // det_spatial_metric
    det_spatial_metric_type& det_spatial_metric_te_temp =
        get<::Tags::TempTensor<23, det_spatial_metric_type>>(vars);

    // inverse_spacetime_metric
    inverse_spacetime_metric_type& inverse_spacetime_metric_te_temp =
        get<::Tags::TempTensor<24, inverse_spacetime_metric_type>>(vars);

    // christoffel_first_kind
    christoffel_first_kind_type& christoffel_first_kind_te_temp =
        get<::Tags::TempTensor<25, christoffel_first_kind_type>>(vars);

    // christoffel_second_kind
    christoffel_second_kind_type& christoffel_second_kind_te_temp =
        get<::Tags::TempTensor<26, christoffel_second_kind_type>>(vars);

    // trace_christoffel
    trace_christoffel_type& trace_christoffel_te_temp =
        get<::Tags::TempTensor<27, trace_christoffel_type>>(vars);

    // normal_spacetime_vector
    normal_spacetime_vector_type& normal_spacetime_vector_te_temp =
        get<::Tags::TempTensor<28, normal_spacetime_vector_type>>(vars);

    // normal_spacetime_one_form
    normal_spacetime_one_form_type& normal_spacetime_one_form_te_temp =
        get<::Tags::TempTensor<29, normal_spacetime_one_form_type>>(vars);

    // da_spacetime_metric
    da_spacetime_metric_type& da_spacetime_metric_te_temp =
        get<::Tags::TempTensor<30, da_spacetime_metric_type>>(vars);

    // d_spacetime_metric
    d_spacetime_metric_type& d_spacetime_metric_te_temp =
        get<::Tags::TempTensor<31, d_spacetime_metric_type>>(vars);
    copy_tensor(d_spacetime_metric, make_not_null(&d_spacetime_metric_te_temp));

    // d_pi
    d_pi_type& d_pi_te_temp = get<::Tags::TempTensor<32, d_pi_type>>(vars);
    copy_tensor(d_pi, make_not_null(&d_pi_te_temp));

    // d_phi
    d_phi_type& d_phi_te_temp = get<::Tags::TempTensor<33, d_phi_type>>(vars);
    copy_tensor(d_phi, make_not_null(&d_phi_te_temp));

    // spacetime_metric
    spacetime_metric_type& spacetime_metric_te_temp =
        get<::Tags::TempTensor<34, spacetime_metric_type>>(vars);
    copy_tensor(spacetime_metric, make_not_null(&spacetime_metric_te_temp));

    // pi
    pi_type& pi_te_temp = get<::Tags::TempTensor<35, pi_type>>(vars);
    copy_tensor(pi, make_not_null(&pi_te_temp));

    // phi
    phi_type& phi_te_temp = get<::Tags::TempTensor<36, phi_type>>(vars);
    copy_tensor(phi, make_not_null(&phi_te_temp));

    // gamma0
    gamma0_type& gamma0_te_temp =
        get<::Tags::TempTensor<37, gamma0_type>>(vars);
    copy_tensor(gamma0, make_not_null(&gamma0_te_temp));

    // gamma1
    gamma1_type& gamma1_te_temp =
        get<::Tags::TempTensor<38, gamma1_type>>(vars);
    copy_tensor(gamma1, make_not_null(&gamma1_te_temp));

    // gamma2
    gamma2_type& gamma2_te_temp =
        get<::Tags::TempTensor<39, gamma2_type>>(vars);
    copy_tensor(gamma2, make_not_null(&gamma2_te_temp));

    // gauge_function
    gauge_function_type& gauge_function_te_temp =
        get<::Tags::TempTensor<40, gauge_function_type>>(vars);
    copy_tensor(gauge_function, make_not_null(&gauge_function_te_temp));

    // spacetime_deriv_gauge_function
    spacetime_deriv_gauge_function_type&
        spacetime_deriv_gauge_function_te_temp =
            get<::Tags::TempTensor<41, spacetime_deriv_gauge_function_type>>(
                vars);
    copy_tensor(spacetime_deriv_gauge_function,
                make_not_null(&spacetime_deriv_gauge_function_te_temp));

    // pi_one_normal_spatial
    pi_one_normal_spatial_type& pi_one_normal_spatial_te_temp =
        get<::Tags::TempTensor<42, pi_one_normal_spatial_type>>(vars);

    // phi_one_normal_spatial
    phi_one_normal_spatial_type& phi_one_normal_spatial_te_temp =
        get<::Tags::TempTensor<43, phi_one_normal_spatial_type>>(vars);

    // Compute TensorExpression result
    BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
        make_not_null(&dt_spacetime_metric_te_temp),
        make_not_null(&dt_pi_te_temp), make_not_null(&dt_phi_te_temp),
        make_not_null(&temp_gamma1_te_temp),
        make_not_null(&temp_gamma2_te_temp),
        make_not_null(&gamma1gamma2_te_temp),
        make_not_null(&pi_two_normals_te_temp),
        make_not_null(&normal_dot_gauge_constraint_te_temp),
        make_not_null(&gamma1_plus_1_te_temp),
        make_not_null(&pi_one_normal_te_temp),
        make_not_null(&gauge_constraint_te_temp),
        make_not_null(&phi_two_normals_te_temp),
        make_not_null(&shift_dot_three_index_constraint_te_temp),
        make_not_null(&phi_one_normal_te_temp), make_not_null(&pi_2_up_te_temp),
        make_not_null(&three_index_constraint_te_temp),
        make_not_null(&phi_1_up_te_temp), make_not_null(&phi_3_up_te_temp),
        make_not_null(&christoffel_first_kind_3_up_te_temp),
        make_not_null(&lapse_te_temp), make_not_null(&shift_te_temp),
        make_not_null(&spatial_metric_te_temp),
        make_not_null(&inverse_spatial_metric_te_temp),
        make_not_null(&det_spatial_metric_te_temp),
        make_not_null(&inverse_spacetime_metric_te_temp),
        make_not_null(&christoffel_first_kind_te_temp),
        make_not_null(&christoffel_second_kind_te_temp),
        make_not_null(&trace_christoffel_te_temp),
        make_not_null(&normal_spacetime_vector_te_temp),
        make_not_null(&normal_spacetime_one_form_te_temp),
        make_not_null(&da_spacetime_metric_te_temp), d_spacetime_metric_te_temp,
        d_pi_te_temp, d_phi_te_temp, spacetime_metric_te_temp, pi_te_temp,
        phi_te_temp, gamma0_te_temp, gamma1_te_temp, gamma2_te_temp,
        gauge_function_te_temp, spacetime_deriv_gauge_function_te_temp,
        make_not_null(&pi_one_normal_spatial_te_temp),
        make_not_null(&phi_one_normal_spatial_te_temp));

    // CHECK christoffel_first_kind (abb)
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        for (size_t c = 0; c < Dim + 1; c++) {
          CHECK_ITERABLE_APPROX(christoffel_first_kind_manual.get(a, b, c),
                                christoffel_first_kind_te_temp.get(a, b, c));
        }
      }
    }

    // CHECK christoffel_second_kind (Abb)
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        for (size_t c = 0; c < Dim + 1; c++) {
          CHECK_ITERABLE_APPROX(christoffel_second_kind_manual.get(a, b, c),
                                christoffel_second_kind_te_temp.get(a, b, c));
        }
      }
    }

    // CHECK trace_christoffel (a)
    for (size_t a = 0; a < Dim + 1; a++) {
      CHECK_ITERABLE_APPROX(trace_christoffel_manual.get(a),
                            trace_christoffel_te_temp.get(a));
    }

    // CHECK gamma1gamma2 (scalar)
    CHECK_ITERABLE_APPROX(gamma1gamma2_manual.get(),
                          gamma1gamma2_te_temp.get());

    // CHECK phi_1_up (Iaa)
    for (size_t i = 0; i < Dim; i++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = 0; b < Dim + 1; b++) {
          CHECK_ITERABLE_APPROX(phi_1_up_manual.get(i, a, b),
                                phi_1_up_te_temp.get(i, a, b));
        }
      }
    }

    // CHECK phi_3_up (iaB)
    for (size_t i = 0; i < Dim; i++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = 0; b < Dim + 1; b++) {
          CHECK_ITERABLE_APPROX(phi_3_up_manual.get(i, a, b),
                                phi_3_up_te_temp.get(i, a, b));
        }
      }
    }

    // CHECK pi_2_up (aB)
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(pi_2_up_manual.get(a, b),
                              pi_2_up_te_temp.get(a, b));
      }
    }

    // CHECK christoffel_first_kind_3_up (abC)
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        for (size_t c = 0; c < Dim + 1; c++) {
          CHECK_ITERABLE_APPROX(
              christoffel_first_kind_3_up_manual.get(a, b, c),
              christoffel_first_kind_3_up_te_temp.get(a, b, c));
        }
      }
    }

    // CHECK pi_one_normal (a)
    for (size_t a = 0; a < Dim + 1; a++) {
      CHECK_ITERABLE_APPROX(pi_one_normal_manual.get(a),
                            pi_one_normal_te_temp.get(a));
    }

    // CHECK pi_one_normal_spatial (i)
    for (size_t i = 0; i < Dim; i++) {
      CHECK_ITERABLE_APPROX(pi_one_normal_spatial_manual.get(i),
                            pi_one_normal_spatial_te_temp.get(i));
    }

    // CHECK pi_two_normals (scalar)
    CHECK_ITERABLE_APPROX(pi_two_normals_manual.get(),
                          pi_two_normals_te_temp.get());

    // CHECK phi_one_normal (ia)
    for (size_t i = 0; i < Dim; i++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        CHECK_ITERABLE_APPROX(phi_one_normal_manual.get(i, a),
                              phi_one_normal_te_temp.get(i, a));
      }
    }

    // CHECK phi_one_normal_spatial (ij)
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        CHECK_ITERABLE_APPROX(phi_one_normal_spatial_manual.get(i, j),
                              phi_one_normal_spatial_te_temp.get(i, j));
      }
    }

    // CHECK phi_two_normals (i)
    for (size_t i = 0; i < Dim; i++) {
      CHECK_ITERABLE_APPROX(phi_two_normals_manual.get(i),
                            phi_two_normals_te_temp.get(i));
    }

    // CHECK three_index_constraint (iaa)
    for (size_t i = 0; i < Dim; i++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = 0; b < Dim + 1; b++) {
          CHECK_ITERABLE_APPROX(three_index_constraint_manual.get(i, a, b),
                                three_index_constraint_te_temp.get(i, a, b));
        }
      }
    }

    // CHECK gauge_constraint (a)
    for (size_t a = 0; a < Dim + 1; a++) {
      CHECK_ITERABLE_APPROX(gauge_constraint_manual.get(a),
                            gauge_constraint_te_temp.get(a));
    }

    // CHECK normal_dot_gauge_constraint (scalar)
    CHECK_ITERABLE_APPROX(normal_dot_gauge_constraint_manual.get(),
                          normal_dot_gauge_constraint_te_temp.get());

    // CHECK gamma1_plus_1 (scalar)
    CHECK_ITERABLE_APPROX(gamma1_plus_1_manual.get(),
                          gamma1_plus_1_te_temp.get());

    // CHECK shift_dot_three_index_constraint (aa)
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(
            shift_dot_three_index_constraint_manual.get(a, b),
            shift_dot_three_index_constraint_te_temp.get(a, b));
      }
    }

    // CHECK dt_spacetime_metric (aa)
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(dt_spacetime_metric_manual.get(a, b),
                              dt_spacetime_metric_te_temp.get(a, b));
      }
    }

    // CHECK dt_pi (aa)
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        CHECK_ITERABLE_APPROX(dt_pi_manual.get(a, b), dt_pi_te_temp.get(a, b));
      }
    }

    // CHECK dt_phi (iaa)
    for (size_t i = 0; i < Dim; i++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = 0; b < Dim + 1; b++) {
          CHECK_ITERABLE_APPROX(dt_phi_manual.get(i, a, b),
                                dt_phi_te_temp.get(i, a, b));
        }
      }
    }
  }
}

template <typename DataType, typename Generator>
void test_benchmarked_implementations(
    const gsl::not_null<Generator*> generator) noexcept {
  test_benchmarked_implementations_core<1, DataType>(generator);
  test_benchmarked_implementations_core<2, DataType>(generator);
  test_benchmarked_implementations_core<3, DataType>(generator);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Benchmark",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_benchmarked_implementations<double>(make_not_null(&generator));
  test_benchmarked_implementations<DataVector>(make_not_null(&generator));
}
