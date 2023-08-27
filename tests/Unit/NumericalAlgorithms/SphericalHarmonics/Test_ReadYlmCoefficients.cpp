// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadYlmCoefficients.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Time, ExpansionCenter_x, ExpansionCenter_y, ExpansionCenter_z, Lmax
// const size_t num_non_coef_columns = 5;

// generate a test Strahlkorper with a random radius function and a given l_max
// and expansion center
template <typename Frame, typename Generator>
Strahlkorper<Frame> generate_test_strahlkorper(
    const gsl::not_null<Generator*> generator,
    const std::uniform_real_distribution<>& distribution, const size_t l_max,
    const std::array<double, 3>& expansion_center) {
  const auto radius = make_with_random_values<DataVector>(
      generator, distribution,
      DataVector(ylm::Spherepack::physical_size(l_max, l_max),
                 std::numeric_limits<double>::signaling_NaN()));
  Strahlkorper<Frame> strahlkorper(
      l_max, l_max, radius, expansion_center,
      StrahlkorperContructorData::RadiusAtCollocationPoints);

  return strahlkorper;
}

template <typename Frame, size_t NumTimes>
std::array<Strahlkorper<Frame>, NumTimes> generate_test_strahlkorpers(
    // const gsl::not_null<std::array<Strahlkorper<Frame>, NumTimes>*>
    // strahlkorpers, const std::string& test_filename, const std::string&
    // subfile_name, const std::array<double, NumTimes> times,
    const std::array<double, 3> expansion_center,
    const std::array<size_t, NumTimes> l_maxes  //,
    /*const size_t max_l*/) {
  //   // Make sure test input can be written in the correct format
  //   constexpr size_t expected_num_coefs = square(max_l + 1);
  //   constexpr size_t expected_num_total_columns =
  //       num_non_coef_columns + expected_num_coefs;
  //   ASSERT(expected_legend.size() == expected_num_total_columns,
  //          "The test Ylm legend expected to be writen does not have the
  //          expected " "number of columns for the given max_l");

  // // Write test ylm data
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 2.0);

  std::array<Strahlkorper<Frame>, NumTimes> strahlkorpers{};
  for (size_t i = 0; i < NumTimes; i++) {
    strahlkorpers[i] = generate_test_strahlkorper<Frame>(
        make_not_null(&generator), distribution, l_maxes[i], expansion_center);
  }

  return strahlkorpers;

  //   std::vector<std::vector<std::string>> legends_to_write(NumTimes);
  //   std::vector<std::vector<double>> data(NumTimes);
  //   for (size_t i = 0; i < NumTimes; i++) {
  //     intrp::callbacks::detail::fill_ylm_legend_and_data(
  //         make_not_null(&(legends[i])), make_not_null(&(data[i])),
  //         strahlkorpers[i], written_times[i], max_l);
  //     ASSERT(legends[i] == expected_legend,
  //       ""
  //     );
  //   }

  // h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);
  // auto& file = test_file.insert<h5::Dat>("/" + subfile_name, legend);
  // file.append(data);
  // test_file.close_current_object();

  //   return strahlkorper;
}

template <typename Frame, size_t NumTimes>
void write_test_strahlkorpers(
    const std::array<Strahlkorper<Frame>, NumTimes>& strahlkorpers,
    const std::string& test_filename, const std::string& subfile_name,
    const std::array<double, NumTimes> times, const size_t max_l) {
  std::vector<std::vector<std::string>> legends(NumTimes);
  std::vector<std::vector<double>> data(NumTimes);
  for (size_t i = 0; i < NumTimes; i++) {
    intrp::callbacks::detail::fill_ylm_legend_and_data(
        make_not_null(&(legends[i])), make_not_null(&(data[i])),
        strahlkorpers[i], times[i], max_l);
  }
  for (size_t i = 0; i < NumTimes - 1; i++) {
    ASSERT(legends[i] == legends[i + 1],
           "Cannot generate test Ylm data to write because the format of the "
           "legend to write is not the same for each row of data to write");
  }

  h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename, true);
  auto& file = test_file.insert<h5::Dat>("/" + subfile_name, legends[0]);
  file.append(data);
  test_file.close_current_object();
}

// bool is_nonnegative_int(const double n) {
//   return n == abs(n) and n == floor(n);
// }

// template <typename Frame, typename Generator>
// void check_writing_format_consistency_impl(
//     const std::vector<std::string> expected_legend,
//     const Strahlkorper<Frame>& strahlkorper, const double time,
//     const size_t max_l) {
//   const size_t expected_total_num_columns =
//       square(max_l + 1) + num_non_coef_columns;
//   ASSERT(expected_legend.size() == expected_total_num_columns,
//          "Expected legend for Ylm test data is not the expected length.");

//   std::vector<std::string> actual_legend;
//   std::vector<double> actual_data;
//   intrp::callbacks::detail::fill_ylm_legend_and_data(
//       make_not_null(&actual_legend), make_not_null(&actual_data),
//       strahlkorper, time, max_l);

//   CHECK(actual_legend == expected_legend);
//   ASSERT(actual_data.size() == actual_legend.size());

//   const std::array<double, 3> expansion_center =
//   strahlkorper.expansion_center(); const double l_max = strahlkorper.l_max();

//   CHECK(actual_data[0] == time);
//   CHECK(actual_data[1] == expansion_center[0]);
//   CHECK(actual_data[2] == expansion_center[1]);
//   CHECK(actual_data[3] == expansion_center[2]);
//   CHECK(actual_data[4] == l_max);

//   for () {

//   }
// }

// // TODO : maybe call in separate test from other stuff
// template <typename Frame, typename Generator>
// void check_writing_format_consistency(
//     const gsl::not_null<Generator*> generator,
//     const std::uniform_real_distribution<>& distribution) {
//   const std::vector<std::string> expected_ylm_legend_without_coefs{
//       "Time", "ExpansionCenter_x", "ExpansionCenter_y", "ExpansionCenter_z",
//       "Lmax"};
//   ASSERT(expected_ylm_legend_without_coefs.size() == num_non_coef_columns,
//          "Test Ylm legend does not contain the expected number of "
//          "non-coefficient columns.");

//   const std::vector<std::string> l_max_2_coef_headers = {
//       "coef(0,0)",  "coef(1,-1)", "coef(1,0)", "coef(1,1)", "coef(2,-2)",
//       "coef(2,-1)", "coef(2,0)",  "coef(2,1)", "coef(2,2)"};

//   std::vector<std::string> l_max_3_coef_headers = l_max_2_coef_headers;
//   l_max_3_coef_headers.push_back("coef(3,-3)");
//   l_max_3_coef_headers.push_back("coef(3,-2)");
//   l_max_3_coef_headers.push_back("coef(3,-1)");
//   l_max_3_coef_headers.push_back("coef(3,0)");
//   l_max_3_coef_headers.push_back("coef(3,1)");
//   l_max_3_coef_headers.push_back("coef(3,2)");
//   l_max_3_coef_headers.push_back("coef(3,3)");

//   const std::array<double, 3> expansion_center{0.3, 0.0, -0.4};
//   const double time = 0.7;
//   const size_t max_l = 3;

//   const auto strahlkorper_lmax_2 = generate_test_strahlkorper<Frame>(
//       make_not_null(&generator), distribution, 2, expansion_center);
//   //   std::vector<std::string> legend_lmax_2;
//   //   std::vector<double> data_lmax_2;
//   //   intrp::callbacks::detail::fill_ylm_legend_and_data(
//   //       make_not_null(&legend_lmax_2), make_not_null(&data_lmax_2),
//   //       strahlkorper_lmax_2, time, max_l);

//   const auto strahlkorper_lmax_3 = generate_test_strahlkorper<Frame>(
//       make_not_null(&generator), distribution, 3, expansion_center);
//   std::vector<std::string> legend_lmax_3;
//   std::vector<double> data_lmax_3;
//   intrp::callbacks::detail::fill_ylm_legend_and_data(
//       make_not_null(&legend_lmax_3), make_not_null(&data_lmax_3),
//       strahlkorper_lmax_3, time, max_l);

//   //   const size_t expected_total_num_columns =
//   //       square(max_l + 1) + num_non_coef_columns;
//   //   ASSERT(is_positive_int());
// }

// test reading in the n last times of data expected to have been written
// (expected_strahlkorpers), where n = num_times_requested
template <typename Frame, size_t NumTimes>
void check_read_ylm_data(
    const std::string& test_filename, const std::string& surface_name,
    const size_t num_times_requested,
    const std::array<Strahlkorper<Frame>, NumTimes>& expected_strahlkorpers) {
  ASSERT(
      NumTimes >= num_times_requested,
      "Requesting to read more rows of Ylm test data than the total number of "
      "rows expected to be written and therefore able to be read.");

  const std::vector<Strahlkorper<Frame>> strahlkorpers =
      ylm::read_ylm_coefficients<Frame>(test_filename, surface_name,
                                        num_times_requested);
  CHECK(strahlkorpers.size() == num_times_requested);

  // check n last times where n = num_times_requested
  for (size_t i = 0, expected_row_number = NumTimes - num_times_requested;
       i < num_times_requested; i++, expected_row_number++) {
    const auto& expected_strahlkorper =
        expected_strahlkorpers[expected_row_number];
    const std::array<double, 3> expected_expansion_center =
        expected_strahlkorper.expansion_center();
    const size_t expected_l_max = expected_strahlkorper.l_max();
    const DataVector& expected_spectral_coefficients =
        expected_strahlkorper.coefficients();
    const size_t expected_spectral_size = expected_spectral_coefficients.size();

    const auto& strahlkorper = strahlkorpers[i];
    CHECK(strahlkorper.expansion_center() == expected_expansion_center);
    CHECK(strahlkorper.l_max() == expected_l_max);
    CHECK(strahlkorper.coefficients().size() == expected_spectral_size);
    CHECK(strahlkorper.coefficients() == expected_spectral_coefficients);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.ReadYlmCoefficients",
                  "[ApparentHorizons][Unit]") {
  // Create a temporary file with test data to read in
  // First, check if the file exists, and delete it if so
  const std::string test_filename{"TestYlmData.h5"};
  if (file_system::check_if_file_exists(test_filename)) {
    file_system::rm(test_filename, true);
  }

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 2.0);

  // first test surface
  using frame_a = Frame::Grid;
  const std::string subfile_name_a = "SurfaceA_Ylm";
  const std::array<double, 3> expansion_center_a{{-0.5, -0.1, 0.3}};
  const size_t max_l_a = 3;

  constexpr size_t number_of_times_a = 3;
  const std::array<double, number_of_times_a> written_times_a{{0.0, 0.1, 0.2}};
  const std::array<size_t, number_of_times_a> l_maxes_a{{3, 2, 3}};

  const auto strahlkorpers_a =
      generate_test_strahlkorpers<frame_a>(expansion_center_a, l_maxes_a);
  write_test_strahlkorpers(strahlkorpers_a, test_filename, subfile_name_a,
                           written_times_a, max_l_a);

  // second test surface
  using frame_b = Frame::Inertial;
  const std::string subfile_name_b = "SurfaceB";
  const std::array<double, 3> expansion_center_b{{0.0, 0.2, -0.6}};
  const size_t max_l_b = 4;

  constexpr size_t number_of_times_b = 1;
  const std::array<double, number_of_times_b> written_times_b{{0.7}};
  const std::array<size_t, number_of_times_b> l_maxes_b{{3}};

  const auto strahlkorpers_b =
      generate_test_strahlkorpers<frame_b>(expansion_center_b, l_maxes_b);
  write_test_strahlkorpers(strahlkorpers_b, test_filename, subfile_name_b,
                           written_times_b, max_l_b);

  check_read_ylm_data<frame_a>(test_filename, subfile_name_a, 1,
                               strahlkorpers_a);
  check_read_ylm_data<frame_a>(test_filename, subfile_name_a, 2,
                               strahlkorpers_a);
  check_read_ylm_data<frame_a>(test_filename, subfile_name_a, 3,
                               strahlkorpers_a);

  check_read_ylm_data<frame_b>(test_filename, subfile_name_b, 1,
                               strahlkorpers_b);

  //   Delete the temporary file created for this test
  //   file_system::rm(test_filename, true);
}
