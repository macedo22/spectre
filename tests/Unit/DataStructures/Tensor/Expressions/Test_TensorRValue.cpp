// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace {
constexpr size_t DIM = 3;

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

template <typename DataType>
auto get_rank2_tensor() {
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<DIM, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<DIM, UpLo::Lo, Frame::Grid>>>
      tensor(3_st);
  assign_unique_values_to_tensor(make_not_null(&tensor));
  std::cout << "made rank 2 tensor in function for testing" << std::endl;
  return tensor;
}

template <typename DataType>
auto get_rank1_lower_tensor() {
  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<DIM, UpLo::Lo, Frame::Grid>>>
      tensor(3_st);
  assign_unique_values_to_tensor(make_not_null(&tensor));
  std::cout << "made rank 1 lower tensor in function for testing" << std::endl;
  return tensor;
}

template <typename DataType>
auto get_rank1_upper_tensor() {
  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<DIM, UpLo::Up, Frame::Grid>>>
      tensor(3_st);
  assign_unique_values_to_tensor(make_not_null(&tensor));
  std::cout << "made rank 1 upper tensor in function for testing" << std::endl;
  return tensor;
}

template <typename DataType>
auto get_rank0_tensor() {
  Tensor<DataType> tensor(3_st);
  std::cout << "made rank 0 tensor in function for testing" << std::endl;
  return tensor;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.SingleTensorRValue",
                  "[DataStructures][Unit]") {
  std::cout << "======= SingleTensorRValue =======\n" << std::endl;
  std::cout << "=== Creating expected Tensor result ===" << std::endl;
  const auto expected_result_tensor = get_rank2_tensor<DataVector>();

  std::cout << "=== Testing destruction of Tensor temporary with evaluation "
               "after creating lvalue expression ==="
            << std::endl;
  std::cout << "Step 1: Create lvalue expression from rvalue Tensor"
            << std::endl;
  // Summary: temporary Tensor destructed before expression evaluation
  // 1. construction of Tensor within function call
  // 2. move construction of Tensor from function return
  // 3. original temp Tensor from (1) destructed
  // 4. call to Tensor::operator() with moved temp Tensor from (2)
  // 5. destruction of moved temp Tensor from (2)
  const auto expr_with_temp_tensor = get_rank2_tensor<DataVector>()(ti_a, ti_b);
  std::cout
      << "Step 2: Evaluate lvalue expression constructed from rvalue Tensor"
      << std::endl;
  // 6. construction of result Tensor, which has empty DataVectors, so fails
  //    CHECKs
  const auto actual_result_tensor_from_lvalue_expr =
      TensorExpressions::evaluate<ti_a, ti_b>(expr_with_temp_tensor);
  for (size_t a = 0; a < DIM + 1; a++) {
    for (size_t b = 0; b < DIM + 1; b++) {
      CHECK(expected_result_tensor.get(a, b) ==
            actual_result_tensor_from_lvalue_expr.get(a, b));
    }
  }

  std::cout
      << "\n=== Testing destruction of Tensor temporary with evaluation of "
         "rvalue expression ==="
      << std::endl;
  // Summary: temporary Tensor destructed after expression evaluation
  // 1. construction of Tensor within function call
  // 2. move construction of Tensor from function return
  // 3. original temp Tensor from (1) destructed
  // 4. call to Tensor::operator() with moved temp Tensor from (2)
  // 5. construction of result Tensor, which is correct and pases CHECKs
  // 6. destruction of moved temp Tensor from (2)
  auto actual_result_tensor_from_rvalue_expr =
      TensorExpressions::evaluate<ti_a, ti_b>(
          get_rank2_tensor<DataVector>()(ti_a, ti_b));
  for (size_t a = 0; a < DIM + 1; a++) {
    for (size_t b = 0; b < DIM + 1; b++) {
      CHECK(expected_result_tensor.get(a, b) ==
            actual_result_tensor_from_rvalue_expr.get(a, b));
    }
  }

  std::cout << "\n======= End of SingleTensorRValue testing ======="
            << std::endl;

  // After test exits:
  // - destruction of actual_result_tensor_from_rvalue_expr
  // - destruction of actual_result_tensor_from_lvalue_expr
  // - destruction of expected_result_tensor
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.NestedTensorRValues",
                  "[DataStructures][Unit]") {
  std::cout << "\n======= NestedTensorRValues =======\n" << std::endl;
  std::cout << "=== Creating Tensor operands to use ===" << std::endl;
  const auto rank2_tensor = get_rank2_tensor<DataVector>();
  const auto rank1_lower_tensor = get_rank1_lower_tensor<DataVector>();
  const auto rank1_upper_tensor = get_rank1_upper_tensor<DataVector>();

  std::cout
      << "\n=== Testing destruction of Tensor temporaries with evaluation "
         "of rvalue expression ==="
      << std::endl;
  // The test expression, where all Tensors used are rvalues:
  //
  //     Tensor_a1(ti_a) + Tensor_ab(ti_a, ti_b) * Tensor_B(ti_B) +
  //     Tensor_a2(ti_a)
  //
  // Tree:
  //
  //                      AddSub
  //                     /       \
  //                 AddSub      TensorAsExpression
  //                /      \                     v
  // TensorAsExpression    TensorContract     Tensor_a2
  //        v                    |
  //    Tensor_a1           OuterProduct
  //                        /          \
  //         TensorAsExpression        TensorAsExpression
  //                v                         v
  //            Tensor_ab                 Tensor_B
  //
  // Summary: temporary Tensor destructed after expression evaluation
  // 1. construction of Tensor_a1 within function call
  // 2. move construction of Tensor_a1 from function return
  // 3. original temp Tensor_a1 from (1) destructed
  // 4. call to Tensor::operator() with moved Tensor_a1 from (2)
  //
  // 5. construction of Tensor_ab within function call
  // 6. move construction of Tensor_ab from function return
  // 7. original temp Tensor_ab from (5) destructed
  // 8. call to Tensor::operator() with moved Tensor_ab from (6)
  //
  // 9. construction of Tensor_B within function call
  // 10. move construction of Tensor_B from function return
  // 11. original temp Tensor_B from (9) destructed
  // 12. call to Tensor::operator() with moved Tensor_B from (10)
  //
  // 13. construction of Tensor_a2 within function call
  // 14. move construction of Tensor_a2 from function return
  // 15. original temp Tensor_a2 from (13) destructed
  // 16. call to Tensor::operator() with moved Tensor_a2 from (14)
  //
  // 17. construction of result Tensor, which is correct and pases CHECKs
  // 18. destruction of Tensor_a2 from (14)
  // 19. destruction of Tensor_B from (10)
  // 20. destruction of Tensor_ab from (6)
  // 21. destruction of Tensor_a1 from (2)
  const auto actual_result_tensor_from_rvalue_expr =
      TensorExpressions::evaluate<ti_a>(
          get_rank1_lower_tensor<DataVector>()(ti_a) +
          get_rank2_tensor<DataVector>()(ti_a, ti_b) *
              get_rank1_upper_tensor<DataVector>()(ti_B) +
          get_rank1_lower_tensor<DataVector>()(ti_a));
  std::cout << "Constructed actual_result_tensor_from_rvalue_expr : "
            << actual_result_tensor_from_rvalue_expr << std::endl;
  DataVector used_for_size =
      DataVector(3, std::numeric_limits<double>::signaling_NaN());
  for (size_t a = 0; a < DIM + 1; a++) {
    DataVector inner_product_sum_over_b =
        make_with_value<DataVector>(used_for_size, 0.0);
    for (size_t b = 0; b < DIM + 1; b++) {
      inner_product_sum_over_b +=
          rank2_tensor.get(a, b) * rank1_upper_tensor.get(b);
    }
    DataVector expected_result_tensor_component = rank1_lower_tensor.get(a) +
                                                  inner_product_sum_over_b +
                                                  rank1_lower_tensor.get(a);
    CHECK(expected_result_tensor_component ==
          actual_result_tensor_from_rvalue_expr.get(a));
  }

  std::cout << "\n======= End of NestedTensorRValues testing ======="
            << std::endl;

  // After test exits:
  // - destruction of actual_result_tensor_from_rvalue_expr
  // - destruction of rank1_upper_tensor
  // - destruction of rank1_lower_tensor
  // - destruction of rank2_tensor
}
