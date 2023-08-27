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
const size_t l_max_column_number = 4;

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
    const std::array<double, 3> expansion_center,
    const std::array<size_t, NumTimes> l_maxes) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 2.0);

  std::array<Strahlkorper<Frame>, NumTimes> strahlkorpers{};
  for (size_t i = 0; i < NumTimes; i++) {
    strahlkorpers[i] = generate_test_strahlkorper<Frame>(
        make_not_null(&generator), distribution, l_maxes[i], expansion_center);
  }

  return strahlkorpers;
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

// Create a temporary file with test data to read in
void write_error_file_and_try_to_read(
    const std::string& filename, const std::string& subfile_name,
    const std::vector<std::string>& legend,
    const std::vector<std::vector<double>>& data,
    const size_t num_times_to_read) {
  h5::H5File<h5::AccessType::ReadWrite> test_file{filename};
  auto& file = test_file.insert<h5::Dat>("/" + subfile_name, legend);
  file.append(data);
  test_file.close_current_object();

  const auto strahlkorpers = ylm::read_ylm_coefficients<Frame::Grid>(
      filename, subfile_name, num_times_to_read);
};

void test_errors() {
  const std::vector<std::string> ylm_legend_without_coefs{
      "Time", "ExpansionCenter_x", "ExpansionCenter_y", "ExpansionCenter_z",
      "Lmax"};

  const std::vector<std::string> l_max_2_coef_headers = {
      "coef(0,0)",  "coef(1,-1)", "coef(1,0)", "coef(1,1)", "coef(2,-2)",
      "coef(2,-1)", "coef(2,0)",  "coef(2,1)", "coef(2,2)"};

  std::vector<std::string> l_max_3_coef_headers = l_max_2_coef_headers;
  l_max_3_coef_headers.push_back("coef(3,-3)");
  l_max_3_coef_headers.push_back("coef(3,-2)");
  l_max_3_coef_headers.push_back("coef(3,-1)");
  l_max_3_coef_headers.push_back("coef(3,0)");
  l_max_3_coef_headers.push_back("coef(3,1)");
  l_max_3_coef_headers.push_back("coef(3,2)");
  l_max_3_coef_headers.push_back("coef(3,3)");

  // construct a legend that is properly formatted
  std::vector<std::string> good_legend = ylm_legend_without_coefs;
  good_legend.insert(good_legend.end(), l_max_3_coef_headers.begin(),
                     l_max_3_coef_headers.end());

  // construct data that is properly formatted
  const std::array<double, 3> times{{0.1, 0.2, 0.3}};
  const std::array<double, 3> l_maxes{{2.0, 3.0, 2.0}};
  std::vector<std::vector<double>> good_data{
      times.size(), std::vector<double>(good_legend.size(), 0.0)};
  good_data[0][l_max_column_number] = l_maxes[0];
  good_data[1][l_max_column_number] = l_maxes[1];
  good_data[2][l_max_column_number] = l_maxes[2];

  const std::string filename{"TestYlmCoefErrors.h5"};
  const std::string subfile_name{std::string{"/Surface_Ylm"}};

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  CHECK_THROWS_WITH(
      ([&filename, &subfile_name, &good_legend]() {
        const std::vector<std::vector<double>> data{};
        write_error_file_and_try_to_read(filename, subfile_name, good_legend,
                                         data, 0);
      }()),
      Catch::Contains("The Ylm data to read from contains 0 rows"));
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  CHECK_THROWS_WITH(
      ([&filename, &subfile_name, &good_legend, &good_data]() {
        write_error_file_and_try_to_read(filename, subfile_name, good_legend,
                                         good_data, 4);
      }()),
      Catch::Contains("The requested number of time values (4) is more than "
                      "the number of rows in the Ylm data"));
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  CHECK_THROWS_WITH(
      ([&filename, &subfile_name, &good_legend, &good_data]() {
        std::vector<std::vector<double>> bad_data = good_data;
        // set an Lmax to a non-integral value
        bad_data[0][l_max_column_number] = 1.2;

        write_error_file_and_try_to_read(filename, subfile_name, good_legend,
                                         bad_data, 3);
      }()),
      Catch::Contains("Row 0 of the Ylm data has an invalid Lmax value"));
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  CHECK_THROWS_WITH(
      ([&filename, &subfile_name, &good_legend, &good_data]() {
        std::vector<std::vector<double>> bad_data = good_data;
        // set an Lmax to a negative value
        bad_data[1][l_max_column_number] = -0.3;

        write_error_file_and_try_to_read(filename, subfile_name, good_legend,
                                         bad_data, 3);
      }()),
      Catch::Contains("Row 1 of the Ylm data has an invalid Lmax value"));
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  CHECK_THROWS_WITH(
      ([&filename, &subfile_name, &good_legend, &good_data]() {
        std::vector<std::vector<double>> bad_data = good_data;
        // set an Lmax that requires more columns of data than given,
        // i.e. good_legend can only hold coefs up to l = 3
        bad_data[2][l_max_column_number] = 4;

        write_error_file_and_try_to_read(filename, subfile_name, good_legend,
                                         bad_data, 3);
      }()),
      Catch::Contains("Row 2 of the Ylm data does not have the expected "
                      "format. For Lmax = 4, expected at least 30 columns"));
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  CHECK_THROWS_WITH(([&filename, &subfile_name, &good_legend, &good_data]() {
                      std::vector<std::vector<double>> bad_data = good_data;
                      // set a non-zero coefficient value for some coef(l,m)
                      // where l > Lmax
                      bad_data[0][good_legend.size() - 1] = 0.8;

                      write_error_file_and_try_to_read(
                          filename, subfile_name, good_legend, bad_data, 3);
                    }()),
                    Catch::Contains("Row 0 of the Ylm data has Lmax 2 but "
                                    "non-zero coefficients for l > Lmax"));
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.ReadYlmCoefficients",
                  "[ApparentHorizons][Unit]") {
  test_errors();

  // Create a temporary file with test data to read in
  // First, check if the file exists, and delete it if so
  const std::string test_filename{"TestYlmCoefs.h5"};
  if (file_system::check_if_file_exists(test_filename)) {
    file_system::rm(test_filename, true);
  }

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

  // Delete the temporary file created for this test
  file_system::rm(test_filename, true);
}
