// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdlib>  // needed for rand, srand, and RAND_MAX?
#include <iterator>
#include <numeric>
#include <time>  // try time.h is this doesn't work
#include <type_traits>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
double get_random_double_in_range(const double min, const double max) {
  return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
}

template <typename... Ts>
void assign_random_values_to_tensor(
    const gsl::not_null<Tensor<double, Ts...>*> tensor, const double min,
    const double max) noexcept {
  for (auto component_it = tensor->begin(); component_it != tensor->end();
       component_it++) {
    *component_it = get_random_double_in_range(min, max);
  }
}

template <typename... Ts>
void assign_random_values_to_tensor(
    const gsl::not_null<Tensor<DataVector, Ts...>*> tensor, const double min,
    const double max) noexcept {
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = get_random_double_in_range(min, max);
    }
  }
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    const gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    const gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

template <typename... Ts>
void zero_initialize_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
  for (auto tensor_it = tensor->begin(); tensor_it != tensor->end();
       tensor_it++) {
    *tensor_it = 0.0;
  }
}

template <typename... Ts>
void zero_initialize_tensor(
    gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = 0.0;
    }
  }
}

constexpr size_t Dim = 3;
constexpr size_t num_grid_points = 5;

using dt_spacetime_metric_type = tnsr::aa<DataVector, Dim>;
using dt_pi_type = tnsr::aa<DataVector, Dim>;
using dt_phi_type = tnsr::iaa<DataVector, Dim>;
using temp_gamma1_type = Scalar<DataVector>;
using temp_2_type = Scalar<DataVector>;
using gamma1gamma2_type = Scalar<DataVector>;
using phi_two_normals_type = Scalar<DataVector>;
using normal_dot_gauge_constraint_type = Scalar<DataVector>;
using gamma1_plus_1_type = Scalar<DataVector>;
using pi_one_normal_type = tnsr::a<DataVector, Dim>;
using gauge_constraint_type = tnsr::a<DataVector, Dim>;
using phi_two_normals_type = tnsr::i<DataVector, Dim>;
using shift_dot_three_index_constraint_type = tnsr::aa<DataVector, Dim>;
using phi_one_normal_type = tnsr::ia<DataVector, Dim>;
using pi_2_up_type = tnsr::aB<DataVector, Dim>;
using three_index_constraint_type = tnsr::iaa<DataVector, Dim>;
using phi_1_up_type = tnsr::Iaa<DataVector, Dim>;
using phi_3_up_type = tnsr::iaB<DataVector, Dim>;
using christoffel_first_kind_3_up_type = tnsr::abC<DataVector, Dim>;
using lapse_type = Scalar<DataVector>;
using shift_type = tnsr::I<DataVector, Dim>;
using spatial_metric_type = tnsr::ii<DataVector, Dim>;
using inverse_spatial_metric_type = tnsr::II<DataVector, Dim>;
using det_spatial_metric_type = Scalar<DataVector>;
using inverse_spacetime_metric_type = tnsr::AA<DataVector, Dim>;
using christoffel_first_kind_type = tnsr::abb<DataVector, Dim>;
using christoffel_second_kind_type = tnsr::Abb<DataVector, Dim>;
using trace_christoffel_type = tnsr::a<DataVector, Dim>;
using normal_spacetime_vector_type = tnsr::A<DataVector, Dim>;
using normal_spacetime_one_form_type = tnsr::a<DataVector, Dim>;
using da_spacetime_metric_type = tnsr::abb<DataVector, Dim>;
using d_spacetime_metric_type = tnsr::iaa<DataVector, Dim>;
using d_pi_type = tnsr::iaa<DataVector, Dim>;
using d_phi_type = tnsr::ijaa<DataVector, Dim>;
using spacetime_metric_type = tnsr::aa<DataVector, Dim>;
using pi_type = tnsr::aa<DataVector, Dim>;
using phi_type = tnsr::iaa<DataVector, Dim>;
using gamm0_type = Scalar<DataVector>;
using gamma1_type = Scalar<DataVector>;
using gamma2_type = Scalar<DataVector>;
using gauge_function_type = tnsr::a<DataVector, Dim>;
using spacetime_deriv_gauge_function_type = tnsr::ab<DataVector, Dim>;

template <size_t Dim>
auto compute_expected_result() noexcept {
  spacetime_metric_type spacetime_metric(num_grid_points);
  assign_random_values_to_tensor(make_not_null(&spacetime_metric), -2, 2);

  spatial_metric_type spatial_metric(num_grid_points);
  gr::spatial_metric(spatial_metric, spacetime_metric);

  determinant_and_inverse(det_spatial_metric, inverse_spatial_metric,
                          *spatial_metric);

  return expected_result;
}

template <typename DataType>
void test_mixed_operations(const DataType& used_for_size) noexcept {
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      R(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&R));

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      S(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&S));

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      G(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&G));

  Tensor<DataType, Symmetry<3, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      H(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&H));

  Tensor<DataType> T{{{used_for_size}}};
  if constexpr (std::is_same_v<DataType, double>) {
    // Replace tensor's value from `used_for_size` with a proper test value
    T.get() = -2.2;
  } else {
    assign_unique_values_to_tensor(make_not_null(&T));
  }

  using result_tensor_type = decltype(G);
  result_tensor_type expected_result_tensor =
      compute_expected_result(R, S, G, H, T, used_for_size);
  // \f$L_{a} = R_{ab}* S^{b} + G_{a} - H_{ba}{}^{b} * T\f$
  result_tensor_type actual_result_tensor_returned =
      TensorExpressions::evaluate<ti_a>(R(ti_a, ti_b) * S(ti_B) + G(ti_a) -
                                        H(ti_b, ti_a, ti_B) * T());
  result_tensor_type actual_result_tensor_filled{};
  TensorExpressions::evaluate<ti_a>(
      make_not_null(&actual_result_tensor_filled),
      R(ti_a, ti_b) * S(ti_B) + G(ti_a) - H(ti_b, ti_a, ti_B) * T());

  for (size_t a = 0; a < 4; a++) {
    CHECK_ITERABLE_APPROX(actual_result_tensor_returned.get(a),
                          expected_result_tensor.get(a));
    CHECK_ITERABLE_APPROX(actual_result_tensor_filled.get(a),
                          expected_result_tensor.get(a));
  }

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    Variables<tmpl::list<::Tags::TempTensor<1, result_tensor_type>>>
        actual_result_tensor_temp_var{used_for_size.size()};
    result_tensor_type& actual_result_tensor_temp =
        get<::Tags::TempTensor<1, result_tensor_type>>(
            actual_result_tensor_temp_var);
    ::TensorExpressions::evaluate<ti_a>(
        make_not_null(&actual_result_tensor_temp),
        R(ti_a, ti_b) * S(ti_B) + G(ti_a) - H(ti_b, ti_a, ti_B) * T());

    for (size_t a = 0; a < 4; a++) {
      CHECK_ITERABLE_APPROX(actual_result_tensor_temp.get(a),
                            expected_result_tensor.get(a));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.GHTimeDerivative",
                  "[DataStructures][Unit]") {
  // srand (time(NULL));
  test_mixed_operations(std::numeric_limits<double>::signaling_NaN());
  test_mixed_operations(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
