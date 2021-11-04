// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename Generator, typename DataType>
void test(const gsl::not_null<Generator*> generator,
          const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  const auto christoffel_second_kind =
      make_with_random_values<tnsr::Ijj<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  const auto d_conformal_christoffel_second_kind =
      make_with_random_values<tnsr::iJkk<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  const auto conformal_spatial_metric =
      make_with_random_values<tnsr::ii<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  const auto inverse_conformal_spatial_metric =
      make_with_random_values<tnsr::II<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  const auto field_d =
      make_with_random_values<tnsr::ijj<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  const auto field_d_up =
      make_with_random_values<tnsr::iJJ<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  const auto field_p =
      make_with_random_values<tnsr::i<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  const auto d_field_p =
      make_with_random_values<tnsr::ij<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);

  // Compiled with clang-10 compile_commands.json command. See
  // compile_command.txt in this directory
  //
  // Before removing inlining of TensorContract::get
  // ------------------------------------------------
  // real    0m40.299s
  // user    0m39.847s
  // sys     0m0.452s
  // ------------------------------------------------
  // After removing inlining of TensorContract::get
  // ------------------------------------------------
  // real    0m10.259s
  // user    0m9.895s
  // sys     0m0.364s
  // ------------------------------------------------
  tnsr::ii<DataType, 3, Frame::Inertial> ricci_tensor{};
  TensorExpressions::evaluate<ti_i, ti_j>(
      make_not_null(&ricci_tensor),
      // Add first terms of \partial_m \Gamma^m_{ij} and
      // -\partial_j \Gamma^m_{im}
      d_conformal_christoffel_second_kind(ti_m, ti_M, ti_i, ti_j) -
          d_conformal_christoffel_second_kind(ti_j, ti_M, ti_i, ti_m) +
          2.0 * ((field_d_up(ti_m, ti_M, ti_L) *
                  (conformal_spatial_metric(ti_j, ti_l) * field_p(ti_i) +
                   conformal_spatial_metric(ti_i, ti_l) * field_p(ti_j) -
                   conformal_spatial_metric(ti_i, ti_j) * field_p(ti_l))) -
                 inverse_conformal_spatial_metric(ti_M, ti_L) *
                     (field_d(ti_m, ti_j, ti_l) * field_p(ti_i) +
                      field_d(ti_m, ti_i, ti_l) * field_p(ti_j) -
                      field_d(ti_m, ti_i, ti_j) * field_p(ti_l)) -
                 (field_d_up(ti_j, ti_M, ti_L) *
                  (conformal_spatial_metric(ti_m, ti_l) * field_p(ti_i) +
                   conformal_spatial_metric(ti_i, ti_l) * field_p(ti_m) -
                   conformal_spatial_metric(ti_i, ti_m) * field_p(ti_l))) +
                 inverse_conformal_spatial_metric(ti_M, ti_L) *
                     (field_d(ti_j, ti_m, ti_l) * field_p(ti_i) +
                      field_d(ti_j, ti_i, ti_l) * field_p(ti_m) -
                      field_d(ti_j, ti_i, ti_m) * field_p(ti_l))) -
          // Add \partial_{(i} P_{j)} type terms
          0.5 * (inverse_conformal_spatial_metric(ti_M, ti_L) *
                 (conformal_spatial_metric(ti_j, ti_l) *
                      (d_field_p(ti_m, ti_i) + d_field_p(ti_i, ti_m)) +
                  conformal_spatial_metric(ti_i, ti_l) *
                      (d_field_p(ti_m, ti_j) + d_field_p(ti_j, ti_m)) -
                  conformal_spatial_metric(ti_i, ti_j) *
                      (d_field_p(ti_m, ti_l) + d_field_p(ti_l, ti_m)) -
                  conformal_spatial_metric(ti_m, ti_l) *
                      (d_field_p(ti_j, ti_i) + d_field_p(ti_i, ti_j)) -
                  conformal_spatial_metric(ti_i, ti_l) *
                      (d_field_p(ti_j, ti_m) + d_field_p(ti_m, ti_j)) +
                  conformal_spatial_metric(ti_i, ti_m) *
                      (d_field_p(ti_j, ti_l) + d_field_p(ti_l, ti_j)))) +
          // Add last two terms for R_{ij}
          christoffel_second_kind(ti_L, ti_i, ti_j) *
              christoffel_second_kind(ti_M, ti_l, ti_m) -
          christoffel_second_kind(ti_L, ti_i, ti_m) *
              christoffel_second_kind(ti_M, ti_l, ti_j));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataVector", "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator),
       DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
