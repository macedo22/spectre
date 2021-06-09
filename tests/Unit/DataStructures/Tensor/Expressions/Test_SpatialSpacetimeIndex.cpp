// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
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
void test_rank4(const DataType& used_for_size) noexcept {
  constexpr size_t dim = 3;
  Tensor<DataType, Symmetry<3, 2, 1, 2>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>>>
      R(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&R));

  // \f$L_{ai} = R_{ai}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<dim, UpLo::Lo, Frame::Grid>>>
      Likaj_from_R_jaik(used_for_size);
  TensorExpressions::evaluate<ti_i, ti_k, ti_a, ti_j>(
      make_not_null(&Likaj_from_R_jaik), R(ti_j, ti_a, ti_i, ti_k));

  for (size_t i = 0; i < dim; i++) {
    for (size_t k = 0; k < dim; k++) {
      for (size_t a = 0; a < dim + 1; a++) {
        for (size_t j = 0; j < dim; j++) {
          CHECK(Likaj_from_R_jaik.get(i, k + 1, a, j) ==
                R.get(j + 1, a, i, k + 1));
        }
      }
    }
  }
}

template <typename DataType, typename Generator>
void test_contractions_rank2(
    const DataType& used_for_size,
    const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  // Contract (spatial, spacetime) tensor
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  const Tensor<DataType> R_contracted =
      TensorExpressions::evaluate(R(ti_I, ti_i));

  // Contract (spacetime, spatial) tensor
  const auto S = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  const Tensor<DataType> S_contracted =
      TensorExpressions::evaluate(S(ti_K, ti_k));

  // Contract (spacetime, spacetime) tensor using generic spatial indices
  const auto T = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>>(
      generator, distribution, used_for_size);
  const Tensor<DataType> T_contracted =
      TensorExpressions::evaluate(T(ti_j, ti_J));

  DataType expected_R_sum = make_with_value<DataType>(used_for_size, 0.0);
  DataType expected_S_sum = make_with_value<DataType>(used_for_size, 0.0);
  DataType expected_T_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    expected_R_sum += R.get(i, i + 1);
    expected_S_sum += S.get(i + 1, i);
    expected_T_sum += T.get(i + 1, i + 1);
  }
  CHECK_ITERABLE_APPROX(R_contracted.get(), expected_R_sum);
  CHECK_ITERABLE_APPROX(S_contracted.get(), expected_S_sum);
  CHECK_ITERABLE_APPROX(T_contracted.get(), expected_T_sum);
}

template <typename DataType, typename Generator>
void test_contractions_rank4(
    const DataType& used_for_size,
    const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                        SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>>(
      generator, distribution, used_for_size);

  // Contract one (spatial, spacetime) pair of indices of a tensor that also
  // takes a generic spatial index for a single non-contracted spacetime index
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      R_contracted_1 =
          TensorExpressions::evaluate<ti_i, ti_K>(R(ti_K, ti_j, ti_i, ti_J));

  for (size_t i = 0; i < 3; i++) {
    for (size_t k = 0; k < 3; k++) {
      DataType expected_R_sum_1 = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t j = 0; j < 3; j++) {
        expected_R_sum_1 += R.get(k, j, i + 1, j + 1);
      }
      CHECK_ITERABLE_APPROX(R_contracted_1.get(i, k), expected_R_sum_1);
    }
  }

  // Contract one (spacetime, spacetime) pair of indices using generic spatial
  // indices and then one (spatial, spatial) pair of indices
  const Tensor<DataType> R_contracted_2 =
      TensorExpressions::evaluate(R(ti_I, ti_i, ti_j, ti_J));

  DataType expected_R_sum_2 = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      expected_R_sum_2 += R.get(i, i, j + 1, j + 1);
    }
  }
  CHECK_ITERABLE_APPROX(R_contracted_2.get(), expected_R_sum_2);

  // Contract two (spatial, spacetime) pairs of indices using generic spatial
  // indices
  const Tensor<DataType> R_contracted_3 =
      TensorExpressions::evaluate(R(ti_I, ti_j, ti_i, ti_J));

  DataType expected_R_sum_3 = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      expected_R_sum_3 += R.get(i, j, i + 1, j + 1);
    }
  }
  CHECK_ITERABLE_APPROX(R_contracted_3.get(), expected_R_sum_3);

  const auto S = make_with_random_values<
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>>(
      generator, distribution, used_for_size);

  // Contract one (spacetime, spacetime) pair of indices using generic spacetime
  // indices and then one (spacetime, spacetime) pair of indices using generic
  // spatial indices
  const Tensor<DataType> S_contracted_1 =
      TensorExpressions::evaluate(S(ti_i, ti_I, ti_a, ti_A));

  DataType expected_S_sum_1 = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    for (size_t a = 0; a < 4; a++) {
      expected_S_sum_1 += S.get(i + 1, i + 1, a, a);
    }
  }
  CHECK_ITERABLE_APPROX(S_contracted_1.get(), expected_S_sum_1);

  // Contract two (spacetime, spacetime) pair of indices using generic spatial
  // indices
  const Tensor<DataType> S_contracted_2 =
      TensorExpressions::evaluate(S(ti_j, ti_I, ti_i, ti_J));

  DataType expected_S_sum_2 = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      expected_S_sum_2 += S.get(j + 1, i + 1, i + 1, j + 1);
    }
  }
  CHECK_ITERABLE_APPROX(S_contracted_2.get(), expected_S_sum_2);
}

template <typename DataType>
void test(const DataType& used_for_size) noexcept {
  test_rhs(used_for_size);
  test_lhs(used_for_size);
  test_both(used_for_size);
  test_rank4(used_for_size);

  MAKE_GENERATOR(generator);
  test_contractions_rank2(used_for_size, make_not_null(&generator));
  test_contractions_rank4(used_for_size, make_not_null(&generator));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.SpatialSpacetimeIndex",
                  "[DataStructures][Unit]") {
  test(std::numeric_limits<double>::signaling_NaN());
  test(DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
