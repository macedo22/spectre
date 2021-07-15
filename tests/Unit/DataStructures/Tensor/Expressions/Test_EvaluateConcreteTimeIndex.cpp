// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// \brief Test evaluation of tensors where concrete time indices are used for
// RHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType, typename Generator>
void test_rhs(const DataType& used_for_size,
              const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // \f$L_{a} = R_{at}\f$ TODO: update this
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      La_from_R_at = TensorExpressions::evaluate<ti_a>(R(ti_a, ti_t));

  for (size_t a = 0; a < dim + 1; a++) {
    CHECK(La_from_R_at.get(a) == R.get(a, 0));
  }
}

// \brief Test evaluation of tensors where concrete time indices are used for
// LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType, typename Generator>
void test_lhs(const DataType& used_for_size,
              const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // \f$L_{ai} = R_{ai}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lat_from_R_a(used_for_size);
  TensorExpressions::evaluate<ti_a, ti_t>(make_not_null(&Lat_from_R_a),
                                          R(ti_a));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t i = 0; i < dim; i++) {
      CHECK(Lat_from_R_a.get(a, 0) == R.get(a));
    }
  }
}

// \brief Test evaluation of tensors where concrete time indices are used for
// RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType, typename Generator>
void test_rhs_and_lhs(const DataType& used_for_size,
                      const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<2, 2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>>>>(
      generator, distribution, used_for_size);

  // TODO: update the latex here and other places in this file
  // \f$L_{aTtb} = R_{tba}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<2, 3, 2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<dim, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>>>
      LaTtb_from_R_tba(used_for_size);
  TensorExpressions::evaluate<ti_a, ti_T, ti_t, ti_b>(
      make_not_null(&LaTtb_from_R_tba), R(ti_t, ti_b, ti_a));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t b = 0; b < dim + 1; b++) {
      CHECK(LaTtb_from_R_tba.get(a, 0, 0, b) == R.get(0, b, a));
    }
  }
}

template <typename DataType>
void test_evaluate_spatial_spacetime_index(
    const DataType& used_for_size) noexcept {
  MAKE_GENERATOR(generator);

  test_rhs(used_for_size, make_not_null(&generator));
  test_lhs(used_for_size, make_not_null(&generator));
  test_rhs_and_lhs(used_for_size, make_not_null(&generator));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateConcreteTimeIndex",
    "[DataStructures][Unit]") {
  test_evaluate_spatial_spacetime_index(
      std::numeric_limits<double>::signaling_NaN());
  test_evaluate_spatial_spacetime_index(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
