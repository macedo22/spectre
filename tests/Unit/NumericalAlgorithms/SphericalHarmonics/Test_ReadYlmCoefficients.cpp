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

template <typename Frame, typename Generator>
Strahlkorper<Frame> generate_strahlkorper(
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

template <typename Frame, size_t NumTimes>
void check_read_ylm_data(
    const std::string& test_filename, const std::string& surface_name,
    const size_t num_times_requested,
    const std::array<Strahlkorper<Frame>, NumTimes>& expected_strahlkorpers) {
  const std::vector<Strahlkorper<Frame>> strahlkorpers =
      ylm::read_ylm_coefficients<Frame>(test_filename, surface_name,
                                        num_times_requested);
  ASSERT(NumTimes >= num_times_requested,
         "Requesting to read more rows of Ylm data than the total number of "
         "rows expected to be written and therefore able to be read.");

  CHECK(strahlkorpers.size() == num_times_requested);

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

    // const size_t expected_spectral_size =
    //     ylm::Spherepack::spectral_size(expected_l_max, expected_l_max);
    // const std::array<double, 3> expected_expansion_center = {
    //     {expected_data[expected_row_number][1],
    //      expected_data[expected_row_number][2],
    //      expected_data[expected_row_number][3]}};

    // const auto& strahlkorper = strahlkorpers[i];
    // CHECK(strahlkorper.l_max() == expected_l_max);
    // CHECK(strahlkorper.expansion_center() == expected_expansion_center);
    // CHECK(strahlkorper.coefficients().size() == expected_spectral_size);

    // const DataVector& spectral_coefficients = strahlkorper.coefficients();
    // size_t h5_data_column = num_non_coef_columns;
    // SpherepackIterator iter(expected_l_max, expected_l_max);
    // for (size_t l = 0; l <= expected_l_max; l++) {
    //   for (int m = -l; m <= static_cast<int>(l); m++) {
    //     iter.set(l, m);
    //     CHECK(spectral_coefficients[iter()] ==
    //           expected_data[expected_row_number][h5_data_column]);
    //     h5_data_column++;
    //   }
    // }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.ReadYlmCoefficients",
                  "[ApparentHorizons][Unit]") {
  //   Strahlkorper<Frame::Inertial> strahlkorper(
  //         3, 3,
  //         get(gr::Solutions::kerr_horizon_radius(
  //             ::ylm::Spherepack(3, 3)
  //                 .theta_phi_points(),
  //             1.0, kerr_horizon.dimensionless_spin)),
  //         kerr_horizon.center,
  //         StrahlkorperContructorData::RadiusAtCollocationPoints);
  //   test_errors();

  // Create a temporary file with test data to read in
  // First, check if the file exists, and delete it if so
  const std::string test_filename{"TestYlmData.h5"};
  //   constexpr uint32_t version_number = 4;
  if (file_system::check_if_file_exists(test_filename)) {
    file_system::rm(test_filename, true);
  }

  // constexpr size_t number_of_times = 3;
  // const std::array<double, number_of_times> written_times{{0.0, 0.1, 0.2}};
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

  std::vector<std::string> l_max_4_coef_headers = l_max_3_coef_headers;
  l_max_4_coef_headers.push_back("coef(4,-4)");
  l_max_4_coef_headers.push_back("coef(4,-3)");
  l_max_4_coef_headers.push_back("coef(4,-2)");
  l_max_4_coef_headers.push_back("coef(4,-1)");
  l_max_4_coef_headers.push_back("coef(4,0)");
  l_max_4_coef_headers.push_back("coef(4,1)");
  l_max_4_coef_headers.push_back("coef(4,2)");
  l_max_4_coef_headers.push_back("coef(4,3)");
  l_max_4_coef_headers.push_back("coef(4,4)");

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-2.0, 2.0);

  // first test surface
  using excision_frame = Frame::Grid;
  const std::string excision_subfile_name = "ExcisionSurface_Ylm";
  const std::array<double, 3> excision_expansion_center{{-0.5, -0.1, 0.3}};
  const size_t excision_max_l = 3;
  std::vector<std::string> excision_legend = ylm_legend_without_coefs;
  excision_legend.insert(excision_legend.end(), l_max_3_coef_headers.begin(),
                         l_max_3_coef_headers.end());

  constexpr size_t excision_number_of_times = 3;
  const std::array<double, excision_number_of_times> excision_written_times{
      {0.0, 0.1, 0.2}};
  std::array<Strahlkorper<excision_frame>, excision_number_of_times>
      excision_strahlkorpers;
  excision_strahlkorpers[0] = generate_strahlkorper<excision_frame>(
      make_not_null(&generator), distribution, 3, excision_expansion_center);
  excision_strahlkorpers[1] = generate_strahlkorper<excision_frame>(
      make_not_null(&generator), distribution, 3, excision_expansion_center);
  excision_strahlkorpers[2] = generate_strahlkorper<excision_frame>(
      make_not_null(&generator), distribution, 3, excision_expansion_center);

  const size_t expected_excision_num_columns =
      square(excision_max_l + 1) + num_non_coef_columns;
  std::vector<std::vector<std::string>> excision_legends(
      excision_number_of_times);
  std::vector<std::vector<double>> excision_data(excision_number_of_times);
  for (size_t i = 0; i < excision_number_of_times; i++) {
    intrp::callbacks::detail::fill_ylm_legend_and_data(
        make_not_null(&(excision_legends[i])),
        make_not_null(&(excision_data[i])), excision_strahlkorpers[i],
        excision_written_times[i], excision_max_l);
    CHECK(excision_legends[i].size() == expected_excision_num_columns);
    CHECK(excision_data[i].size() == expected_excision_num_columns);
  }

  // second test surface
  using horizon_frame = Frame::Inertial;
  const std::string horizon_subfile_name = "ApparentHorizon_Ylm";
  const std::array<double, 3> horizon_expansion_center{{0.0, 0.2, -0.6}};
  const size_t horizon_max_l = 4;
  std::vector<std::string> horizon_legend = ylm_legend_without_coefs;
  horizon_legend.insert(horizon_legend.end(), l_max_4_coef_headers.begin(),
                        l_max_4_coef_headers.end());

  constexpr size_t horizon_number_of_times = 1;
  const std::array<double, horizon_number_of_times> horizon_written_times{
      {0.7}};
  std::array<Strahlkorper<horizon_frame>, horizon_number_of_times>
      horizon_strahlkorpers;
  horizon_strahlkorpers[0] = generate_strahlkorper<horizon_frame>(
      make_not_null(&generator), distribution, 3, horizon_expansion_center);

  const size_t expected_horizon_num_columns =
      square(horizon_max_l + 1) + num_non_coef_columns;
  std::vector<std::vector<std::string>> horizon_legends(
      horizon_number_of_times);
  std::vector<std::vector<double>> horizon_data(horizon_number_of_times);
  for (size_t i = 0; i < horizon_number_of_times; i++) {
    intrp::callbacks::detail::fill_ylm_legend_and_data(
        make_not_null(&(horizon_legends[i])), make_not_null(&(horizon_data[i])),
        horizon_strahlkorpers[i], horizon_written_times[i], horizon_max_l);
    CHECK(horizon_legends[i].size() == expected_horizon_num_columns);
    CHECK(horizon_data[i].size() == expected_horizon_num_columns);
  }

  h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);
  auto& excision_file =
      test_file.insert<h5::Dat>("/" + excision_subfile_name, excision_legend);
  excision_file.append(excision_data);
  test_file.close_current_object();
  auto& horizon_file =
      test_file.insert<h5::Dat>("/" + horizon_subfile_name, horizon_legend);
  horizon_file.append(horizon_data);
  test_file.close_current_object();

  check_read_ylm_data<excision_frame>(test_filename, excision_subfile_name, 3,
                                      excision_strahlkorpers);

  check_read_ylm_data<horizon_frame>(test_filename, horizon_subfile_name, 1,
                                     horizon_strahlkorpers);

  //   Delete the temporary file created for this test
  //   file_system::rm(test_filename, true);
}
