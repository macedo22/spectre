// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
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
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Time, ExpansionCenter_x, ExpansionCenter_y, ExpansionCenter_z, Lmax
const size_t num_non_coef_columns = 5;

// generate test Ylm data with a given l_max, times, and expansion center and
// random Ylm coefficients
template <size_t Lmax, size_t NumberOfTimes, typename Generator>
std::vector<std::vector<double>> generate_surface_ylm_data(
    const gsl::not_null<Generator*> generator,
    const std::uniform_real_distribution<>& distribution,
    const std::array<double, NumberOfTimes>& times,
    const std::array<double, 3>& expansion_center) {
  const size_t num_coefs = square(Lmax + 1);
  const size_t num_columns = num_non_coef_columns + num_coefs;
  std::vector<std::vector<double>> data(NumberOfTimes,
                                        std::vector<double>(num_columns));

  for (size_t i = 0; i < NumberOfTimes; i++) {
    data[i][0] = times[0];
    data[i][1] = expansion_center[0];
    data[i][2] = expansion_center[1];
    data[i][3] = expansion_center[2];
    data[i][4] = Lmax;

    const auto coefs = make_with_random_values<std::array<double, num_coefs>>(
        generator, distribution, 0.0);

    for (size_t j = 0; j < num_coefs; j++) {
      data[i][j + num_non_coef_columns] = coefs[j];
    }
  }
  return data;
};

template <typename Frame>
void check_read_ylm_data(const std::string& test_filename,
                         const std::string& surface_name,
                         const std::vector<std::vector<double>>& expected_data,
                         const size_t num_times_requested) {
  // Make sure input expected_data is formatted correctly
  const size_t total_num_times = expected_data.size();
  REQUIRE(total_num_times > 0);
  ASSERT(expected_data[0][4] == abs(expected_data[0][4]) and
             expected_data[0][4] == floor(expected_data[0][4]),
         "Ylm test Lmax is not a positive integer value");
  const size_t expected_l_max = static_cast<size_t>(expected_data[0][4]);
  const std::string columns_error_message =
      "Ylm test data does not have the expected number of columns based on its "
      "Lmax value";
  const size_t expected_num_columns = expected_data[0].size();
  ASSERT(
      expected_num_columns == square(expected_l_max + 1) + num_non_coef_columns,
      columns_error_message);
  for (size_t i = 1; i < total_num_times; i++) {
    ASSERT(expected_data[0][4] == expected_data[i][4],
           "Ylm test data should have the same Lmax for each row for a given "
           "surface.");
    ASSERT(expected_num_columns == expected_data[i].size(),
           columns_error_message);
  }

  // Call function to test and check against expected data
  const std::vector<Strahlkorper<Frame>> strahlkorpers =
      ylm::read_ylm_coefficients<Frame>(test_filename, surface_name,
                                        num_times_requested);
  CHECK(strahlkorpers.size() == num_times_requested);

  const size_t expected_spectral_size =
      ylm::Spherepack::spectral_size(expected_l_max, expected_l_max);

  for (size_t i = 0,
              expected_row_number = total_num_times - num_times_requested;
       i < num_times_requested; i++, expected_row_number++) {
    const std::array<double, 3> expected_expansion_center = {
        {expected_data[expected_row_number][1],
         expected_data[expected_row_number][2],
         expected_data[expected_row_number][3]}};

    const auto& strahlkorper = strahlkorpers[i];
    CHECK(strahlkorper.l_max() == expected_l_max);
    CHECK(strahlkorper.expansion_center() == expected_expansion_center);
    CHECK(strahlkorper.coefficients().size() == expected_spectral_size);

    const DataVector spectral_coefficients = strahlkorper.coefficients();
    size_t h5_data_column = num_non_coef_columns;
    SpherepackIterator iter(expected_l_max, expected_l_max);
    for (size_t l = 0; l <= expected_l_max; l++) {
      for (int m = -l; m <= static_cast<int>(l); m++) {
        iter.set(l, m);
        CHECK(spectral_coefficients[iter()] ==
              expected_data[expected_row_number][h5_data_column]);
        h5_data_column++;
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.ReadYlmCoefficients",
                  "[ApparentHorizons][Unit]") {
  //   test_errors();

  // Create a temporary file with test data to read in
  // First, check if the file exists, and delete it if so
  const std::string test_filename{"TestYlmData.h5"};
  //   constexpr uint32_t version_number = 4;
  if (file_system::check_if_file_exists(test_filename)) {
    file_system::rm(test_filename, true);
  }

  constexpr size_t number_of_times = 3;
  const std::array<double, number_of_times> written_times{{0.0, 0.1, 0.2}};
  const std::vector<std::string> ylm_legend_without_coefs{
      "Time", "ExpansionCenter_x", "ExpansionCenter_y", "ExpansionCenter_z",
      "Lmax"};
  ASSERT(ylm_legend_without_coefs.size() == num_non_coef_columns,
         "Test Ylm legend does not contain the expected number of "
         "non-coefficient columns.");

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

  // first test surface
  const std::string excision_name = "ExcisionSurface";
  const std::array<double, 3> excision_expansion_center{{-0.5, -0.1, 0.3}};
  const size_t excision_l_max = 2;
  std::vector<std::string> excision_legend = ylm_legend_without_coefs;
  excision_legend.insert(excision_legend.end(), l_max_2_coef_headers.begin(),
                         l_max_2_coef_headers.end());

  // second test surface
  const std::string horizon_name = "ApparentHorizon";
  const std::array<double, 3> horizon_expansion_center{{0.0, 0.2, -0.6}};
  const size_t horizon_l_max = 3;
  std::vector<std::string> horizon_legend = ylm_legend_without_coefs;
  horizon_legend.insert(horizon_legend.end(), l_max_3_coef_headers.begin(),
                        l_max_3_coef_headers.end());

  // fill ylm data and random ylm coefs
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 2.0);

  const std::vector<std::vector<double>> excision_data =
      generate_surface_ylm_data<excision_l_max>(make_not_null(&generator),
                                                distribution, written_times,
                                                excision_expansion_center);
  const std::vector<std::vector<double>> horizon_data =
      generate_surface_ylm_data<horizon_l_max>(make_not_null(&generator),
                                               distribution, written_times,
                                               horizon_expansion_center);

  h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);
  auto& excision_file =
      test_file.insert<h5::Dat>("/" + excision_name + "_Ylm", excision_legend);
  excision_file.append(excision_data);
  test_file.close_current_object();
  auto& horizon_file =
      test_file.insert<h5::Dat>("/" + horizon_name + "_Ylm", horizon_legend);
  horizon_file.append(horizon_data);
  test_file.close_current_object();

  check_read_ylm_data<Frame::Inertial>(test_filename, excision_name,
                                       excision_data, 3);

  check_read_ylm_data<Frame::Grid>(test_filename, horizon_name, horizon_data,
                                   1);

  //   Delete the temporary file created for this test
  //   file_system::rm(test_filename, true);
}
