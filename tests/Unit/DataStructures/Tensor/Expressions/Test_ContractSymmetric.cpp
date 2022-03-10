// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <climits>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
constexpr size_t Dim = 3;

template <typename R_type, typename S_type,
          typename DataType = typename R_type::type>
Scalar<DataType> compute_expected_3x3(const R_type& R, const S_type& S) {
  auto result = make_with_value<Scalar<DataType>>(get<0, 0, 0>(R), 0.0);

  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      for (size_t k = 0; k < Dim; k++) {
        get(result) += R.get(i, j, k) * S.get(i, j, k);
      }
    }
  }

  return result;
}

template <typename A_type, typename B_type,
          typename DataType = typename A_type::type>
Scalar<DataType> compute_expected_4x4(const A_type& A, const B_type& B) {
  auto result = make_with_value<Scalar<DataType>>(get<0, 0, 0, 0>(A), 0.0);

  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      for (size_t k = 0; k < Dim; k++) {
        for (size_t l = 0; l < Dim; l++) {
          get(result) += A.get(i, j, k, l) * B.get(i, j, k, l);
        }
      }
    }
  }

  return result;
}

template <typename Generator, typename DataType>
void test(const gsl::not_null<Generator*> generator,
          const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 2.0);

  // upper <1, 1>
  const auto G =
      make_with_random_values<tnsr::II<DataType, Dim, Frame::Inertial>>(
          generator, distribution, used_for_size);
  // lower <1, 1>
  const auto H =
      make_with_random_values<tnsr::ii<DataType, Dim, Frame::Inertial>>(
          generator, distribution, used_for_size);
  // upper <1, 1, 1>
  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  // lower <1, 1, 1>
  const auto S = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  // upper <1, 1, 1, 1>
  const auto A = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  // lower <1, 1, 1, 1>
  const auto B = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  // lower <2, 2, 1, 1>
  const auto C = make_with_random_values<
      Tensor<DataType, Symmetry<2, 2, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // symmetric 3x3
  const auto expected_3x3_result = compute_expected_3x3(R, S);

  const size_t num_ind_comp_3x3 = decltype(R)::structure::size();
  const std::array<size_t, num_ind_comp_3x3> multipliers_3x3 = {1, 3, 3, 3, 6,
                                                                3, 1, 3, 3, 1};
  size_t current_index_3x3 = 0;

  auto actual_3x3_result =
      make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = i; j < Dim; j++) {
      for (size_t k = j; k < Dim; k++) {
        get(actual_3x3_result) += gsl::at(multipliers_3x3, current_index_3x3) *
                                  R.get(i, j, k) * S.get(i, j, k);
        current_index_3x3++;
      }
    }
  }

  CHECK_ITERABLE_APPROX(actual_3x3_result, expected_3x3_result);

  // symmetric 4x4
  const auto expected_4x4_result = compute_expected_4x4(A, B);

  const size_t num_ind_comp_4x4 = decltype(A)::structure::size();
  const std::array<size_t, num_ind_comp_4x4> multipliers_4x4 = {
      1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1};
  size_t current_index_4x4 = 0;

  auto actual_4x4_result =
      make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = i; j < Dim; j++) {
      for (size_t k = j; k < Dim; k++) {
        for (size_t l = k; l < Dim; l++) {
          get(actual_4x4_result) +=
              gsl::at(multipliers_4x4, current_index_4x4) * A.get(i, j, k, l) *
              B.get(i, j, k, l);
          current_index_4x4++;
        }
      }
    }
  }

  CHECK_ITERABLE_APPROX(actual_4x4_result, expected_4x4_result);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Contract",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test(make_not_null(&generator), std::numeric_limits<double>::signaling_NaN());
  //   test_contractions(
  //       make_not_null(&generator),
  //       DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
