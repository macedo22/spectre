// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"

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

// \brief Test the outer product of a rank 0, rank 1, and rank 2 tensor is
// correctly evaluated
//
// \details
// The outer product cases tested are permutations of the form:
// - \f$L^{a}{}_{ib} = R * S^{a} * T_{bi}\f$
//
// Each case represents an ordering for the operands and the LHS indices.
//
// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_rhs(const DataType& used_for_size) noexcept {
  constexpr size_t dim = 3;
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      R(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&R));

  // \f$L_{ai} = R_{ai}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_ai = TensorExpressions::evaluate<ti_a, ti_i>(R(ti_a, ti_i));

  // \f$L_{ia} = R_{ai}\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<dim, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_ai = TensorExpressions::evaluate<ti_i, ti_a>(R(ti_a, ti_i));

  // \f$L_{ai} = R_{ia}\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_ia = TensorExpressions::evaluate<ti_a, ti_i>(R(ti_i, ti_a));

  // \f$L_{ia} = R_{ia}\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<dim, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_ia = TensorExpressions::evaluate<ti_i, ti_a>(R(ti_i, ti_a));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t i = 0; i < dim; i++) {
      // std::cout << "(a, i) : (" << a << ", " << i << ")" << std::endl;
      CHECK(Lai_from_R_ai.get(a, i) == R.get(a, i + 1));
      CHECK(Lia_from_R_ai.get(i, a) == R.get(a, i + 1));
      CHECK(Lai_from_R_ia.get(a, i) == R.get(i + 1, a));
      CHECK(Lia_from_R_ia.get(i, a) == R.get(i + 1, a));
    }
  }
}

// \brief Test the outer product of a rank 0, rank 1, and rank 2 tensor is
// correctly evaluated
//
// \details
// The outer product cases tested are permutations of the form:
// - \f$L^{a}{}_{ib} = R * S^{a} * T_{bi}\f$
//
// Each case represents an ordering for the operands and the LHS indices.
//
// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_lhs(const DataType& used_for_size) noexcept {
  constexpr size_t dim = 3;
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<dim, UpLo::Lo, Frame::Inertial>>>
      R(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&R));

  // \f$L_{ai} = R_{ai}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_ai(used_for_size);
  TensorExpressions::evaluate<ti_a, ti_i>(make_not_null(&Lai_from_R_ai),
                                          R(ti_a, ti_i));

  // \f$L_{ia} = R_{ai}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_ai(used_for_size);
  TensorExpressions::evaluate<ti_i, ti_a>(make_not_null(&Lia_from_R_ai),
                                          R(ti_a, ti_i));

  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpatialIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      S(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&S));

  // \f$L_{ia} = S_{ia}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_S_ia(used_for_size);
  TensorExpressions::evaluate<ti_i, ti_a>(make_not_null(&Lia_from_S_ia),
                                          S(ti_i, ti_a));

  // \f$L_{ai} = S_{ia}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_S_ia(used_for_size);
  TensorExpressions::evaluate<ti_a, ti_i>(make_not_null(&Lai_from_S_ia),
                                          S(ti_i, ti_a));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t i = 0; i < dim; i++) {
      // std::cout << "(a, i) : (" << a << ", " << i << ")" << std::endl;
      CHECK(Lai_from_R_ai.get(a, i + 1) == R.get(a, i));
      CHECK(Lia_from_R_ai.get(i + 1, a) == R.get(a, i));
      CHECK(Lia_from_S_ia.get(i + 1, a) == S.get(i, a));
      CHECK(Lai_from_S_ia.get(a, i + 1) == S.get(i, a));
    }
  }
}

// \brief Test the outer product of a rank 0, rank 1, and rank 2 tensor is
// correctly evaluated
//
// \details
// The outer product cases tested are permutations of the form:
// - \f$L^{a}{}_{ib} = R * S^{a} * T_{bi}\f$
//
// Each case represents an ordering for the operands and the LHS indices.
//
// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_both(const DataType& used_for_size) noexcept {
  constexpr size_t dim = 3;
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      R(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&R));

  // \f$L_{ai} = R_{ai}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_ai(used_for_size);
  TensorExpressions::evaluate<ti_a, ti_i>(make_not_null(&Lai_from_R_ai),
                                          R(ti_a, ti_i));

  // \f$L_{ia} = R_{ai}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_ai(used_for_size);
  TensorExpressions::evaluate<ti_i, ti_a>(make_not_null(&Lia_from_R_ai),
                                          R(ti_a, ti_i));

  // \f$L_{ai} = R_{ia}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_ia(used_for_size);
  TensorExpressions::evaluate<ti_a, ti_i>(make_not_null(&Lai_from_R_ia),
                                          R(ti_i, ti_a));

  // \f$L_{ia} = R_{ia}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_ia(used_for_size);
  TensorExpressions::evaluate<ti_i, ti_a>(make_not_null(&Lia_from_R_ia),
                                          R(ti_i, ti_a));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t i = 0; i < dim; i++) {
      // std::cout << "(a, i) : (" << a << ", " << i << ")" << std::endl;
      CHECK(Lai_from_R_ai.get(a, i + 1) == R.get(a, i + 1));
      CHECK(Lia_from_R_ai.get(i + 1, a) == R.get(a, i + 1));
      CHECK(Lai_from_R_ia.get(a, i + 1) == R.get(i + 1, a));
      CHECK(Lia_from_R_ia.get(i + 1, a) == R.get(i + 1, a));
    }
  }
}

template <typename DataType>
void test(const DataType& used_for_size) noexcept {
  test_rhs(used_for_size);
  test_lhs(used_for_size);
  test_both(used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.SpatialSpacetimeIndex",
                  "[DataStructures][Unit]") {
  test(std::numeric_limits<double>::signaling_NaN());
  test(DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
