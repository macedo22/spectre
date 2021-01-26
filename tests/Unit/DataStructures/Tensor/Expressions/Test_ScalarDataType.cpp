// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"

namespace {
template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

// template <class...T>
// struct td;

template <typename DataType,
          typename DecayedDataType = typename std::decay_t<DataType>>
void test_addsub_scalar_datatype(
    DataType&& scalar, const Tensor<DecayedDataType>& tensor) noexcept {
  // td<decltype(scalar)> idk;
  // using decayed_datatype = typename std::decay<DataType>::type;
  // const Tensor<decayed_datatype> R{{{6.0}}};
  // Tensor<DataType> T{{{used_for_size}}};
  // if constexpr (std::is_same_v<DataType, double>) {
  //   // Replace tensor's value from `used_for_size` with a proper test value
  //   T.get() = -2.2;
  // } else {
  //   assign_unique_values_to_tensor(make_not_null(&T));
  // }

  //   const auto R_S_sum_expr = R() + std::forward<decltype(scalar)>(scalar);
  //   // <- this one const Tensor<double> R_S_sum_1 =
  //   TensorExpressions::evaluate(R_S_sum_expr); // <- this one

  //   const Tensor<DataType> R_S_sum_1 = TensorExpressions::evaluate(R() +
  //   std::forward<DataType&&>(scalar));
  const Tensor<DecayedDataType> R_S_sum_1 =
      TensorExpressions::evaluate(  // <- this
          tensor() +
          std::forward<DataType>(scalar));  // <- this one  // <- this
  // const Tensor<DataType> R_S_sum_1 = TensorExpressions::evaluate(R() +
  // scalar);

  //   // Contract (upper, lower) tensor
  //   // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  //   // return type of `evaluate`
  //   Tensor<DataType, Symmetry<2, 1>,
  //          index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
  //                     SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
  //       Rul(used_for_size);
  //   assign_unique_values_to_tensor(make_not_null(&Rul));

  //   const auto RIi_expr = Rul(ti_I, ti_i);
  //   const Tensor<DataType> RIi_contracted =
  //   TensorExpressions::evaluate(RIi_expr);

  //   DataType expected_RIi_sum = make_with_value<DataType>(used_for_size,
  //   0.0); for (size_t i = 0; i < 3; i++) {
  //     expected_RIi_sum += Rul.get(i, i);

  //     const std::array<size_t, 2> expected_uncontracted_lhs_tensor_index{{i,
  //     i}}; CHECK(RIi_expr.get_tensor_index_to_sum({{}}, i) ==
  //           expected_uncontracted_lhs_tensor_index);
  //   }
  //   CHECK(RIi_contracted.get() == expected_RIi_sum);

  //   // Contract (lower, upper) tensor
  //   Tensor<DataType, Symmetry<2, 1>,
  //          index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
  //                     SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
  //       Rlu(used_for_size);
  //   assign_unique_values_to_tensor(make_not_null(&Rlu));

  //   const auto RgG_expr = Rlu(ti_g, ti_G);
  //   const Tensor<DataType> RgG_contracted =
  //   TensorExpressions::evaluate(RgG_expr);

  //   DataType expected_RgG_sum = make_with_value<DataType>(used_for_size,
  //   0.0); for (size_t g = 0; g < 4; g++) {
  //     expected_RgG_sum += Rlu.get(g, g);

  //     const std::array<size_t, 2> expected_uncontracted_lhs_tensor_index{{g,
  //     g}}; CHECK(RgG_expr.get_tensor_index_to_sum({{}}, g) ==
  //           expected_uncontracted_lhs_tensor_index);
  //   }
  //   CHECK(RgG_contracted.get() == expected_RgG_sum);
}

template <typename DataType,
          typename DecayedDataType = typename std::decay<DataType>::type>
void test_tensor_plus_scalar(DataType&& scalar,
                             const Tensor<DecayedDataType>& tensor) noexcept {
  //std::cout << "test_tensor_plus_scalar" << std::endl;
  const DecayedDataType expected_sum = tensor.get() + scalar;
  const Tensor<DecayedDataType> actual_sum =
      TensorExpressions::evaluate(tensor() + std::forward<DataType>(scalar));
  // std::cout << "actual_sum tensor : " << actual_sum << std::endl;
  CHECK(actual_sum.get() == expected_sum);
}

template <typename DataType,
          typename DecayedDataType = typename std::decay<DataType>::type>
void test_scalar_plus_tensor(DataType&& scalar,
                             const Tensor<DecayedDataType>& tensor) noexcept {
  //std::cout << "test_scalar_plus_tensor" << std::endl;
  const DecayedDataType expected_sum = scalar + tensor.get();
  const Tensor<DecayedDataType> actual_sum =
      TensorExpressions::evaluate(std::forward<DataType>(scalar) + tensor());
  CHECK(actual_sum.get() == expected_sum);
}

template <typename DataType,
          typename DecayedDataType = typename std::decay<DataType>::type>
void test_tensor_minus_scalar(DataType&& scalar,
                              const Tensor<DecayedDataType>& tensor) noexcept {
  const DecayedDataType expected_difference = tensor.get() - scalar;
  const Tensor<DecayedDataType> actual_difference =
      TensorExpressions::evaluate(tensor() - std::forward<DataType>(scalar));
  CHECK(actual_difference.get() == expected_difference);
}

template <typename DataType,
          typename DecayedDataType = typename std::decay<DataType>::type>
void test_scalar_minus_tensor(DataType&& scalar,
                              const Tensor<DecayedDataType>& tensor) noexcept {
  const DecayedDataType expected_difference = scalar - tensor.get();
  const Tensor<DecayedDataType> actual_difference =
      TensorExpressions::evaluate(std::forward<DataType>(scalar) - tensor());
  CHECK(actual_difference.get() == expected_difference);
}

template <typename DataType>
void test_addsub_scalar_lvalue(const DataType& scalar,
                               const Tensor<DataType>& tensor) noexcept {
  test_tensor_plus_scalar(scalar, tensor);
  test_scalar_plus_tensor(scalar, tensor);
  test_tensor_minus_scalar(scalar, tensor);
  test_scalar_minus_tensor(scalar, tensor);
}

void test_addsub_scalar_rvalue(const Tensor<double>& tensor) noexcept {
  test_tensor_plus_scalar(-2.5, tensor);
  test_scalar_plus_tensor(0.8, tensor);
  test_tensor_minus_scalar(1.2, tensor);
  test_scalar_minus_tensor(3.4, tensor);
}

void test_addsub_scalar_rvalue(const Tensor<DataVector>& tensor) noexcept {
  test_tensor_plus_scalar(DataVector{2.0, -1.1, 12.4}, tensor);
  test_scalar_plus_tensor(DataVector{-7.2, 4.9, 0.0}, tensor);
  test_tensor_minus_scalar(DataVector{0.5, -2.7, 3.6}, tensor);
  test_scalar_minus_tensor(DataVector{0.0, 9.2, -0.7}, tensor);
}

template <typename T, typename DataType,
          typename DecayedDataType = typename std::decay<DataType>::type>
void test_tensor_times_scalar(DataType&& scalar, const T& tensor) noexcept {
  T expected_product{};
  for (size_t a = 0; a < tmpl::at_c<typename T::index_list, 0>::dim; a++) {
    for (size_t i = 0; i < tmpl::at_c<typename T::index_list, 1>::dim; i++) {
      expected_product.get(a, i) = tensor.get(a, i) * scalar;
    }
  }
  const T actual_product = TensorExpressions::evaluate<ti_a, ti_I>(
      tensor(ti_a, ti_I) * std::forward<DataType>(scalar));
  for (size_t a = 0; a < tmpl::at_c<typename T::index_list, 0>::dim; a++) {
    for (size_t i = 0; i < tmpl::at_c<typename T::index_list, 1>::dim; i++) {
      CHECK(actual_product.get(a, i) == expected_product.get(a, i));
    }
  }
}

template <typename T, typename DataType,
          typename DecayedDataType = typename std::decay<DataType>::type>
void test_scalar_times_tensor(DataType&& scalar, const T& tensor) noexcept {
  T expected_product{};
  for (size_t a = 0; a < tmpl::at_c<typename T::index_list, 0>::dim; a++) {
    for (size_t i = 0; i < tmpl::at_c<typename T::index_list, 1>::dim; i++) {
      expected_product.get(a, i) = scalar * tensor.get(a, i);
    }
  }
  const T actual_product = TensorExpressions::evaluate<ti_a, ti_I>(
      std::forward<DataType>(scalar) * tensor(ti_a, ti_I));
  for (size_t a = 0; a < tmpl::at_c<typename T::index_list, 0>::dim; a++) {
    for (size_t i = 0; i < tmpl::at_c<typename T::index_list, 1>::dim; i++) {
      CHECK(actual_product.get(a, i) == expected_product.get(a, i));
    }
  }
}

template <typename T, typename DataType>
void test_product_scalar_lvalue(const DataType& scalar,
                                const T& tensor) noexcept {
  test_tensor_times_scalar(scalar, tensor);
  test_scalar_times_tensor(scalar, tensor);
}

template <typename T,
          Requires<std::is_same_v<typename T::type, double>> = nullptr>
void test_product_scalar_rvalue(const T& tensor) noexcept {
  test_tensor_times_scalar(-6.7, tensor);
  test_scalar_times_tensor(0.9, tensor);
}

template <typename T,
          Requires<std::is_same_v<typename T::type, DataVector>> = nullptr>
void test_product_scalar_rvalue(const T& tensor) noexcept {
  test_tensor_times_scalar(DataVector{6.1, -5.2, 0.0}, tensor);
  test_scalar_times_tensor(DataVector{-8.4, 0.0, 4.7}, tensor);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.ScalarDataType",
                  "[DataStructures][Unit]") {
  // put back below 4 lines
  const Tensor<double> addsub_test_tensor1{{{7.4}}};
  const double addsub_test_scalar1 = 8.2;
  test_addsub_scalar_lvalue(addsub_test_scalar1, addsub_test_tensor1);
  test_addsub_scalar_rvalue(addsub_test_tensor1);

  // auto expr = 5000.0 + addsub_test_tensor1() + 10000.0;

  // const double expected_sum = 7.4 + 2.5;
  // const Tensor<double> actual_sum =
  //     TensorExpressions::evaluate(addsub_test_tensor1() + 2.5);
  // CHECK(actual_sum.get() == expected_sum);

  // put back below 4 lines
  const Tensor<DataVector> addsub_test_tensor2{
      {{DataVector{12.3, -1.1, -2.4}}}};
  const DataVector addsub_test_scalar2{0.0, -7.8, 6.9};
  test_addsub_scalar_lvalue(addsub_test_scalar2, addsub_test_tensor2);
  test_addsub_scalar_rvalue(addsub_test_tensor2);

  //   auto expr = addsub_test_tensor2() + DataVector{1.0, 2.0, 3.0} +
  //   addsub_test_tensor2(); auto res = TensorExpressions::evaluate(expr);

  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      product_test_tensor1(1_st);
  assign_unique_values_to_tensor(make_not_null(&product_test_tensor1));
  const double product_test_scalar1 = 2.3;
  test_product_scalar_lvalue(product_test_scalar1, product_test_tensor1);
  test_product_scalar_rvalue(product_test_tensor1);

  Tensor<DataVector, Symmetry<2, 1>,
         index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      product_test_tensor2(3_st);
  assign_unique_values_to_tensor(make_not_null(&product_test_tensor2));
  const DataVector product_test_scalar2{1.1, 0.0, -3.9};
  test_product_scalar_lvalue(product_test_scalar2, product_test_tensor2);
  test_product_scalar_rvalue(product_test_tensor2);

  // test_tensor_plus_scalar(-2.5, tensor1);
  // test_scalar_plus_tensor(0.8, tensor1);
  // test_tensor_minus_scalar(1.2, tensor1);
  // test_scalar_minus_tensor(3.4, tensor1);
  // const double scalar1 = 8.2;
  // test_tensor_plus_scalar(scalar1, tensor1);
  // test_scalar_plus_tensor(scalar1, tensor1);
  // test_tensor_minus_scalar(scalar1, tensor1);
  // test_scalar_minus_tensor(scalar1, tensor1);
  // test_addsub_scalar_lvalue(scalar1, tensor1);

  // // const DataVector dv{12.3, -1.1, -2.4};
  // const Tensor<DataVector> tensor2{{{DataVector{12.3, -1.1, -2.4} /*dv*/}}};
  // // DataVector&& dv = {-2.1, 0.0, -5.6};
  // test_tensor_plus_scalar(DataVector{-2.1, 0.0, -5.6}/*std::move(dv)*/,
  // tensor2); const DataVector scalar2{0.0, -7.8, 6.9};
  // test_tensor_plus_scalar(scalar2, tensor2);

  // test_addsub_scalar_datatype(DataVector{-2.1, 0.0, 5.6});
  //   const double x = 8.2;
  //   const double& ref = x;
  //   test_addsub_scalar_datatype(ref);

  //   // Test adding and subtracting rank 0 tensors
  //   const double R_value = 2.5;
  //   const double S_value = -1.25;
  //   const Tensor<double> R{{{R_value}}};
  //   const Tensor<double> S{{{S_value}}};
  //   const double expected_R_S_sum = R_value + S_value;
  //   const double expected_R_S_difference = R_value - S_value;

  //   const Tensor<double> R_S_sum_1 = TensorExpressions::evaluate(R() + S());
  //   const Tensor<double> R_S_sum_2 = TensorExpressions::evaluate(R() +
  //   S_value); const Tensor<double> R_S_sum_3 =
  //   TensorExpressions::evaluate(R_value + S()); const Tensor<double>
  //   R_S_difference_1 =
  //       TensorExpressions::evaluate(R() - S());
  //   const Tensor<double> R_S_difference_2 =
  //       TensorExpressions::evaluate(R() - S_value);
  //   const Tensor<double> R_S_difference_3 =
  //       TensorExpressions::evaluate(R_value - S());

  //   CHECK(R_S_sum_1.get() == expected_R_S_sum);
  //   CHECK(R_S_sum_2.get() == expected_R_S_sum);
  //   CHECK(R_S_sum_3.get() == expected_R_S_sum);
  //   CHECK(R_S_difference_1.get() == expected_R_S_difference);
  //   CHECK(R_S_difference_2.get() == expected_R_S_difference);
  //   CHECK(R_S_difference_3.get() == expected_R_S_difference);

  //   Tensor<DataVector> dv_tensor(DataVector{-6.2, 1.4, 2.5});
  //   std::cout << "dv_tensor : " << dv_tensor << std::endl;
  //   const Tensor<DataVector> dv_te =
  //       TensorExpressions::evaluate(DataVector{1.0, -2.0, 4.0} +
  //       dv_tensor());
  //   CHECK(DataVector{1.0, -2.0, 4.0} + dv_tensor.get() == dv_te.get());

  //   // Test adding and subtracting a scalar with a contraction
  //   Tensor<double, Symmetry<2, 1>,
  //          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
  //                     SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
  //       Tul{};
  //   std::iota(Tul.begin(), Tul.end(), 0.0);

  //   double expected_trace = 0.0;
  //   for (size_t a = 0; a < 4; a++) {
  //     expected_trace += Tul.get(a, a);
  //   }
  //   const double expected_R_TAa_sum = R_value + expected_trace;
  //   const double expected_R_TAa_difference_1 = R_value - expected_trace;
  //   const double expected_R_TAa_difference_2 = expected_trace - R_value;

  //   const Tensor<double> scalar_with_contraction_sum_1 =
  //       TensorExpressions::evaluate(R() + Tul(ti_A, ti_a));
  //   const Tensor<double> scalar_with_contraction_sum_2 =
  //       TensorExpressions::evaluate(Tul(ti_A, ti_a) + R());
  //   const Tensor<double> scalar_with_contraction_sum_3 =
  //       TensorExpressions::evaluate(R_value + Tul(ti_A, ti_a));
  //   const Tensor<double> scalar_with_contraction_sum_4 =
  //       TensorExpressions::evaluate(Tul(ti_A, ti_a) + R_value);
  //   const Tensor<double> scalar_with_contraction_difference_1 =
  //       TensorExpressions::evaluate(R() - Tul(ti_A, ti_a));
  //   const Tensor<double> scalar_with_contraction_difference_2 =
  //       TensorExpressions::evaluate(Tul(ti_A, ti_a) - R());
  //   const Tensor<double> scalar_with_contraction_difference_3 =
  //       TensorExpressions::evaluate(R_value - Tul(ti_A, ti_a));
  //   const Tensor<double> scalar_with_contraction_difference_4 =
  //       TensorExpressions::evaluate(Tul(ti_A, ti_a) - R_value);

  //   CHECK(scalar_with_contraction_sum_1.get() == expected_R_TAa_sum);
  //   CHECK(scalar_with_contraction_sum_2.get() == expected_R_TAa_sum);
  //   CHECK(scalar_with_contraction_sum_3.get() == expected_R_TAa_sum);
  //   CHECK(scalar_with_contraction_sum_4.get() == expected_R_TAa_sum);
  //   CHECK(scalar_with_contraction_difference_1.get() ==
  //         expected_R_TAa_difference_1);
  //   CHECK(scalar_with_contraction_difference_2.get() ==
  //         expected_R_TAa_difference_2);
  //   CHECK(scalar_with_contraction_difference_3.get() ==
  //         expected_R_TAa_difference_1);
  //   CHECK(scalar_with_contraction_difference_4.get() ==
  //         expected_R_TAa_difference_2);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubtractRank0",
                  "[DataStructures][Unit]") {
  //   // Test adding and subtracting rank 0 tensors
  //   const double R_value = 2.5;
  //   const double S_value = -1.25;
  //   const Tensor<double> R{{{R_value}}};
  //   const Tensor<double> S{{{S_value}}};
  //   const double expected_R_S_sum = R_value + S_value;
  //   const double expected_R_S_difference = R_value - S_value;

  //   const Tensor<double> R_S_sum_1 = TensorExpressions::evaluate(R() + S());
  //   const Tensor<double> R_S_sum_2 = TensorExpressions::evaluate(R() +
  //   S_value); const Tensor<double> R_S_sum_3 =
  //   TensorExpressions::evaluate(R_value + S()); const Tensor<double>
  //   R_S_difference_1 =
  //       TensorExpressions::evaluate(R() - S());
  //   const Tensor<double> R_S_difference_2 =
  //       TensorExpressions::evaluate(R() - S_value);
  //   const Tensor<double> R_S_difference_3 =
  //       TensorExpressions::evaluate(R_value - S());

  //   CHECK(R_S_sum_1.get() == expected_R_S_sum);
  //   CHECK(R_S_sum_2.get() == expected_R_S_sum);
  //   CHECK(R_S_sum_3.get() == expected_R_S_sum);
  //   CHECK(R_S_difference_1.get() == expected_R_S_difference);
  //   CHECK(R_S_difference_2.get() == expected_R_S_difference);
  //   CHECK(R_S_difference_3.get() == expected_R_S_difference);

  //   Tensor<DataVector> dv_tensor(DataVector{-6.2, 1.4, 2.5});
  //   std::cout << "dv_tensor : " << dv_tensor << std::endl;
  //   const Tensor<DataVector> dv_te =
  //       TensorExpressions::evaluate(DataVector{1.0, -2.0, 4.0} +
  //       dv_tensor());
  //   CHECK(DataVector{1.0, -2.0, 4.0} + dv_tensor.get() == dv_te.get());

  //   // Test adding and subtracting a scalar with a contraction
  //   Tensor<double, Symmetry<2, 1>,
  //          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
  //                     SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
  //       Tul{};
  //   std::iota(Tul.begin(), Tul.end(), 0.0);

  //   double expected_trace = 0.0;
  //   for (size_t a = 0; a < 4; a++) {
  //     expected_trace += Tul.get(a, a);
  //   }
  //   const double expected_R_TAa_sum = R_value + expected_trace;
  //   const double expected_R_TAa_difference_1 = R_value - expected_trace;
  //   const double expected_R_TAa_difference_2 = expected_trace - R_value;

  //   const Tensor<double> scalar_with_contraction_sum_1 =
  //       TensorExpressions::evaluate(R() + Tul(ti_A, ti_a));
  //   const Tensor<double> scalar_with_contraction_sum_2 =
  //       TensorExpressions::evaluate(Tul(ti_A, ti_a) + R());
  //   const Tensor<double> scalar_with_contraction_sum_3 =
  //       TensorExpressions::evaluate(R_value + Tul(ti_A, ti_a));
  //   const Tensor<double> scalar_with_contraction_sum_4 =
  //       TensorExpressions::evaluate(Tul(ti_A, ti_a) + R_value);
  //   const Tensor<double> scalar_with_contraction_difference_1 =
  //       TensorExpressions::evaluate(R() - Tul(ti_A, ti_a));
  //   const Tensor<double> scalar_with_contraction_difference_2 =
  //       TensorExpressions::evaluate(Tul(ti_A, ti_a) - R());
  //   const Tensor<double> scalar_with_contraction_difference_3 =
  //       TensorExpressions::evaluate(R_value - Tul(ti_A, ti_a));
  //   const Tensor<double> scalar_with_contraction_difference_4 =
  //       TensorExpressions::evaluate(Tul(ti_A, ti_a) - R_value);

  //   CHECK(scalar_with_contraction_sum_1.get() == expected_R_TAa_sum);
  //   CHECK(scalar_with_contraction_sum_2.get() == expected_R_TAa_sum);
  //   CHECK(scalar_with_contraction_sum_3.get() == expected_R_TAa_sum);
  //   CHECK(scalar_with_contraction_sum_4.get() == expected_R_TAa_sum);
  //   CHECK(scalar_with_contraction_difference_1.get() ==
  //         expected_R_TAa_difference_1);
  //   CHECK(scalar_with_contraction_difference_2.get() ==
  //         expected_R_TAa_difference_2);
  //   CHECK(scalar_with_contraction_difference_3.get() ==
  //         expected_R_TAa_difference_1);
  //   CHECK(scalar_with_contraction_difference_4.get() ==
  //         expected_R_TAa_difference_2);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubtractContract",
                  "[DataStructures][Unit]") {
    // Test adding and subtracting a scalar with a contraction
    // Tensor<double, Symmetry<2, 1>,
    //        index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
    //                   SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
    //     Tul{};
    // std::iota(Tul.begin(), Tul.end(), 0.0);

    // double expected_trace = 0.0;
    // for (size_t a = 0; a < 4; a++) {
    //   expected_trace += Tul.get(a, a);
    // }
    // const double expected_R_TAa_sum = R_value + expected_trace;
    // const double expected_R_TAa_difference_1 = R_value - expected_trace;
    // const double expected_R_TAa_difference_2 = expected_trace - R_value;

    // const Tensor<double> scalar_with_contraction_sum_1 =
    //     TensorExpressions::evaluate(R() + Tul(ti_A, ti_a));
    // const Tensor<double> scalar_with_contraction_sum_2 =
    //     TensorExpressions::evaluate(Tul(ti_A, ti_a) + R());
    // const Tensor<double> scalar_with_contraction_sum_3 =
    //     TensorExpressions::evaluate(R_value + Tul(ti_A, ti_a));
    // const Tensor<double> scalar_with_contraction_sum_4 =
    //     TensorExpressions::evaluate(Tul(ti_A, ti_a) + R_value);
    // const Tensor<double> scalar_with_contraction_difference_1 =
    //     TensorExpressions::evaluate(R() - Tul(ti_A, ti_a));
    // const Tensor<double> scalar_with_contraction_difference_2 =
    //     TensorExpressions::evaluate(Tul(ti_A, ti_a) - R());
    // const Tensor<double> scalar_with_contraction_difference_3 =
    //     TensorExpressions::evaluate(R_value - Tul(ti_A, ti_a));
    // const Tensor<double> scalar_with_contraction_difference_4 =
    //     TensorExpressions::evaluate(Tul(ti_A, ti_a) - R_value);

    // CHECK(scalar_with_contraction_sum_1.get() == expected_R_TAa_sum);
    // CHECK(scalar_with_contraction_sum_2.get() == expected_R_TAa_sum);
    // CHECK(scalar_with_contraction_sum_3.get() == expected_R_TAa_sum);
    // CHECK(scalar_with_contraction_sum_4.get() == expected_R_TAa_sum);
    // CHECK(scalar_with_contraction_difference_1.get() ==
    //       expected_R_TAa_difference_1);
    // CHECK(scalar_with_contraction_difference_2.get() ==
    //       expected_R_TAa_difference_2);
    // CHECK(scalar_with_contraction_difference_3.get() ==
    //       expected_R_TAa_difference_1);
    // CHECK(scalar_with_contraction_difference_4.get() ==
    //       expected_R_TAa_difference_2);
}
