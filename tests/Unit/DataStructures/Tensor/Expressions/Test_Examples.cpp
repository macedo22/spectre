// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <complex>
#include <cstddef>
#include <iostream>
#include <random>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename Generator, typename DataType>
void test_evaluate(const gsl::not_null<Generator*> generator,
                   const std::uniform_real_distribution<>& distribution,
                   const DataType& used_for_size) {
  constexpr size_t Dim = 3;

  const auto R = make_with_random_values<tnsr::ab<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto g = make_with_random_values<tnsr::AA<DataType, Dim>>(
      generator, distribution, used_for_size);

  auto expected_product =
      make_with_value<tnsr::Ab<DataType, Dim>>(used_for_size, 0.0);
  for (size_t c = 0; c < Dim + 1; c++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        expected_product.get(c, b) += R.get(a, b) * g.get(a, c);
      }
    }
  }

  {
    auto R_up =
        tenex::evaluate<ti::C, ti::b>(R(ti::a, ti::b) * g(ti::A, ti::C));
    CHECK_ITERABLE_APPROX(R_up, expected_product);
  }
  {
    tnsr::Ab<DataType, Dim> R_up{};
    tenex::evaluate<ti::C, ti::b>(make_not_null(&R_up),
                                  R(ti::a, ti::b) * g(ti::A, ti::C));
    CHECK_ITERABLE_APPROX(R_up, expected_product);
  }
}

template <typename Generator, typename DataType>
void test_basic_operations(const gsl::not_null<Generator*> generator,
                           const std::uniform_real_distribution<>& distribution,
                           const DataType& used_for_size) {
  constexpr size_t Dim = 3;

  const auto R = make_with_random_values<tnsr::ab<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto S = make_with_random_values<tnsr::ab<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto T = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto U = make_with_random_values<tnsr::Ab<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto V = make_with_random_values<tnsr::aBC<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto G = make_with_random_values<tnsr::a<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto H = make_with_random_values<tnsr::A<DataType, Dim>>(
      generator, distribution, used_for_size);

  // addition
  {
    auto L = tenex::evaluate<ti::a, ti::b>(R(ti::a, ti::b) + S(ti::b, ti::a));

    tnsr::ab<DataType, Dim> expected_result{};
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        expected_result.get(a, b) = R.get(a, b) + S.get(b, a);
      }
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  // subtraction
  {
    auto L = tenex::evaluate(1.0 - T());

    const Scalar<DataType> expected_result{1.0 - get(T)};

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  // contraction of a single tensor
  {
    auto L = tenex::evaluate(U(ti::A, ti::a));

    auto expected_result =
        make_with_value<Scalar<DataType>>(used_for_size, 0.0);
    for (size_t a = 0; a < Dim + 1; a++) {
      get(expected_result) += U.get(a, a);
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    auto L = tenex::evaluate<ti::B>(V(ti::a, ti::B, ti::A));

    auto expected_result =
        make_with_value<tnsr::A<DataType, Dim>>(used_for_size, 0.0);
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        expected_result.get(b) += V.get(a, b, a);
      }
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  // inner and outer products
  {
    auto L = tenex::evaluate(G(ti::a) * H(ti::A));

    auto expected_result =
        make_with_value<Scalar<DataType>>(used_for_size, 0.0);
    for (size_t a = 0; a < Dim + 1; a++) {
      get(expected_result) += G.get(a) * H.get(a);
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    auto L = tenex::evaluate<ti::c, ti::b>(T() * G(ti::a) * G(ti::c) *
                                           U(ti::A, ti::b));

    auto expected_result =
        make_with_value<tnsr::ab<DataType, Dim>>(used_for_size, 0.0);
    for (size_t c = 0; c < Dim + 1; c++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        for (size_t a = 0; a < Dim + 1; a++) {
          expected_result.get(c, b) += G.get(a) * G.get(c) * U.get(a, b);
        }
        expected_result.get(c, b) *= get(T);
      }
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  // divison
  {
    auto L = tenex::evaluate<ti::a>(G(ti::a) / 2.0);

    tnsr::a<DataType, Dim> expected_result{};
    for (size_t a = 0; a < Dim + 1; a++) {
      expected_result.get(a) = G.get(a) / 2.0;
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    auto L = tenex::evaluate<ti::b, ti::a>(R(ti::a, ti::b) / T());

    tnsr::ab<DataType, Dim> expected_result{};
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        expected_result.get(b, a) = R.get(a, b) / get(T);
      }
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  // square root
  {
    auto L = tenex::evaluate(sqrt(T()));

    const Scalar<DataType> expected_result{sqrt(get(T))};

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    auto L = tenex::evaluate(sqrt(G(ti::a) * H(ti::A)));

    auto expected_result =
        make_with_value<Scalar<DataType>>(used_for_size, 0.0);
    for (size_t a = 0; a < Dim + 1; a++) {
      get(expected_result) += G.get(a) * H.get(a);
    }
    get(expected_result) = sqrt(get(expected_result));

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
}

tnsr::aa<double, 3> compute_expected_specify_lhs_symmetry(
    const tnsr::a<double, 3>& R) {
  tnsr::aa<double, 3> L{};
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = a; b < 4; b++) {
      L.get(a, b) = R.get(a) * R.get(b);
    }
  }
  return L;
}

void test_specify_lhs_symmetry() {
  {
    tnsr::a<double, 3> R{{1.0, 2.0, 3.0}};
    auto L = tenex::evaluate<ti::a, ti::b>(R(ti::a) * R(ti::b));
    static_assert(std::is_same_v<decltype(L), tnsr::ab<double, 3>>);

    const tnsr::aa<double, 3> expected_result =
        compute_expected_specify_lhs_symmetry(R);
    for (size_t a = 0; a < 4; a++) {
      for (size_t b = 0; b < 4; b++) {
        CHECK(L.get(a, b) == expected_result.get(a, b));
        CHECK(L.get(a, b) == expected_result.get(b, a));
      }
    }
  }
  {
    tnsr::a<double, 3> R{};
    tnsr::aa<double, 3> L{};
    tenex::evaluate<ti::a, ti::b>(make_not_null(&L), R(ti::a) * R(ti::b));

    const tnsr::aa<double, 3> expected_result =
        compute_expected_specify_lhs_symmetry(R);
    for (size_t a = 0; a < 4; a++) {
      for (size_t b = a; b < 4; b++) {
        CHECK(L.get(a, b) == expected_result.get(a, b));
      }
    }
  }
}

void test_assign_number() {
  {
    tnsr::ab<double, 3> L{};
    tenex::evaluate<ti::a, ti::b>(make_not_null(&L), -1.0);

    const auto expected_result = make_with_value<tnsr::ab<double, 3>>(
        std::numeric_limits<double>::signaling_NaN(), -1.0);
    CHECK(L == expected_result);
  }
  {
    // construct LHS tensor with size 5 DataVector
    tnsr::ab<DataVector, 3> L{DataVector(0.0, 5)};
    tenex::evaluate<ti::a, ti::b>(make_not_null(&L), -1.0);

    const size_t num_points = L[0].size();
    const auto expected_result = make_with_value<tnsr::ab<DataVector, 3>>(
        DataVector(num_points, std::numeric_limits<double>::signaling_NaN()),
        -1.0);
    CHECK(L == expected_result);
  }
}

template <typename Generator, typename DataType>
void test_spatial_and_time_indices(
    const gsl::not_null<Generator*> generator,
    const std::uniform_real_distribution<>& distribution,
    const DataType& used_for_size) {
  constexpr size_t Dim = 3;

  const auto inverse_spatial_metric =
      make_with_random_values<tnsr::II<DataType, Dim>>(generator, distribution,
                                                       used_for_size);
  const auto spacetime_metric =
      make_with_random_values<tnsr::aa<DataType, Dim>>(generator, distribution,
                                                       used_for_size);

  auto lapse = tenex::evaluate(sqrt(inverse_spatial_metric(ti::I, ti::J) *
                                        spacetime_metric(ti::j, ti::t) *
                                        spacetime_metric(ti::i, ti::t) -
                                    spacetime_metric(ti::t, ti::t)));

  auto expected_result = make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      get(expected_result) += inverse_spatial_metric.get(i, j) *
                              spacetime_metric.get(j + 1, 0) *
                              spacetime_metric.get(i + 1, 0);
    }
  }
  get(expected_result) -= get<0, 0>(spacetime_metric);
  get(expected_result) = sqrt(get(expected_result));

  CHECK_ITERABLE_APPROX(lapse, expected_result);
}

template <typename Generator>
void test_assign_component_subsets(
    const gsl::not_null<Generator*> generator,
    const std::uniform_real_distribution<>& distribution,
    const DataVector& used_for_size) {
  constexpr size_t Dim = 3;

  const auto spatial_metric =
      make_with_random_values<tnsr::ii<DataVector, Dim>>(
          generator, distribution, used_for_size);
  const auto shift = make_with_random_values<tnsr::I<DataVector, Dim>>(
      generator, distribution, used_for_size);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      generator, distribution, used_for_size);

  tnsr::aa<DataVector, Dim> spacetime_metric{};
  tenex::evaluate<ti::t, ti::t>(
      make_not_null(&spacetime_metric),
      -lapse() * lapse() +
          shift(ti::M) * shift(ti::N) * spatial_metric(ti::m, ti::n));
  tenex::evaluate<ti::t, ti::i>(make_not_null(&spacetime_metric),
                                spatial_metric(ti::m, ti::i) * shift(ti::M));
  tenex::evaluate<ti::i, ti::j>(make_not_null(&spacetime_metric),
                                spatial_metric(ti::i, ti::j));

  const DataVector lapse_squared = square(get(lapse));

  // note: there are more efficient ways to implement this equation, but
  // choosing the simplest to read and write
  tnsr::aa<DataVector, 3> expected_result{used_for_size};
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = i; j < Dim; j++) {
      expected_result.get(i + 1, j + 1) = spatial_metric.get(i, j);
    }
  }

  for (size_t i = 0; i < Dim; i++) {
    expected_result.get(0, i + 1) = spatial_metric.get(0, i) * shift.get(0);
    for (size_t m = 1; m < Dim; m++) {
      expected_result.get(0, i + 1) += spatial_metric.get(m, i) * shift.get(m);
    }
  }

  get<0, 0>(expected_result) = -lapse_squared;
  for (size_t m = 0; m < Dim; m++) {
    for (size_t n = 0; n < Dim; n++) {
      get<0, 0>(expected_result) +=
          spatial_metric.get(m, n) * shift.get(m) * shift.get(n);
    }
  }

  CHECK_ITERABLE_APPROX(spacetime_metric, expected_result);
}

// void test_examples() {

// }
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Examples",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator, 17);
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const double number_used_for_size =
      std::numeric_limits<double>::signaling_NaN();
  const DataVector vector_used_for_size =
      DataVector(1, std::numeric_limits<double>::signaling_NaN());

  test_evaluate(make_not_null(&generator), distribution, vector_used_for_size);
  test_basic_operations(make_not_null(&generator), distribution,
                        number_used_for_size);
  test_specify_lhs_symmetry();
  test_assign_number();
  test_spatial_and_time_indices(make_not_null(&generator), distribution,
                                number_used_for_size);
  test_assign_component_subsets(make_not_null(&generator), distribution,
                                vector_used_for_size);

  //   test_evaluate(make_not_null(&generator), distribution,
  //                 std::numeric_limits<double>::signaling_NaN());
  //   TestHelpers::tenex::Examples::test_mixed_operations(
  //       make_not_null(&generator),
  //       std::complex<double>(std::numeric_limits<double>::signaling_NaN(),
  //                            std::numeric_limits<double>::signaling_NaN()));
  //   TestHelpers::tenex::Examples::test_mixed_operations(
  //       make_not_null(&generator),
  //       DataVector(5, std::numeric_limits<double>::signaling_NaN()));
  //   TestHelpers::tenex::Examples::test_mixed_operations(
  //       make_not_null(&generator),
  //       ComplexDataVector(5, std::numeric_limits<double>::signaling_NaN()));

  // TODO : to add:
  // - psi4 for demoing complex datavector use with std::complex:
  // https://spectre-code.org/group__GeneralRelativityGroup.html#ga57dde0a2811628294312038d28cbb383
  //
  // - lapse for demoing time and spatial indices for RHS spacetime:
  //   https://spectre-code.org/group__GeneralRelativityGroup.html#gaf6dbe3d6807eb2fd55bf5fefceb79698
  //   note: already an exmaple above, but maybe use thios because it's shorter
}
