// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
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

  h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);

  constexpr size_t number_of_times = 3;
  const std::array<double, number_of_times> expected_times{{0.0, 0.1, 0.2}};
  const std::array<double, number_of_times> expansion_center{{-0.5, -0.1, 0.3}};
  const std::array<std::string, 2> target_names{
      {"ExcisionSurface", "ApparentHorizon"}};

  const size_t l_max = 2;
  const size_t num_non_coef_columns = 5;
  const size_t num_coef_columns = square(l_max + 1);
  const size_t num_columns = num_coef_columns + num_non_coef_columns;

  const std::vector<std::string> ylm_legend{"Time",
                                            "ExpansionCenter_x",
                                            "ExpansionCenter_y",
                                            "ExpansionCenter_z",
                                            "Lmax",
                                            "coef(0,0)",
                                            "coef(1,-1)",
                                            "coef(1,0)",
                                            "coef(1,1)",
                                            "coef(2,-2)",
                                            "coef(2,-1)",
                                            "coef(2,0)",
                                            "coef(2,1)",
                                            "coef(2,2)"};

  std::vector<std::vector<double>> excision_data(
      number_of_times, std::vector<double>(num_columns));
  std::vector<std::vector<double>> horizon_data(
      number_of_times, std::vector<double>(num_columns));

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 2.0);

  for (size_t i = 0; i < number_of_times; i++) {
    excision_data[i][0] = expected_times[0];
    excision_data[i][1] = expansion_center[0];
    excision_data[i][2] = expansion_center[1];
    excision_data[i][3] = expansion_center[2];
    excision_data[i][4] = l_max;

    horizon_data[i][0] = expected_times[0];
    horizon_data[i][1] = expansion_center[0];
    horizon_data[i][2] = expansion_center[1];
    horizon_data[i][3] = expansion_center[2];
    horizon_data[i][4] = l_max;

    const std::array<double, num_coef_columns> excision_coefs =
        make_with_random_values<std::array<double, num_coef_columns>>(
            make_not_null(&generator), distribution, 0.0);
    const std::array<double, num_coef_columns> horizon_coefs =
        make_with_random_values<std::array<double, num_coef_columns>>(
            make_not_null(&generator), distribution, 0.0);

    for (size_t j = 0; j < num_coef_columns; j++) {
      excision_data[i][j + num_non_coef_columns] = excision_coefs[j];
      horizon_data[i][j + num_non_coef_columns] = horizon_coefs[j];
    }
  }

  test_file.close_current_object();
  auto& excision_file =
      test_file.insert<h5::Dat>("/" + target_names[0] + "_Ylm", ylm_legend);
  excision_file.append(excision_data);
  test_file.close_current_object();
  auto& horizon_file =
      test_file.insert<h5::Dat>("/" + target_names[1] + "_Ylm", ylm_legend);
  horizon_file.append(horizon_data);
  test_file.close_current_object();

  const size_t num_requested_excision_strahlkorpers = 3;
  const std::vector<Strahlkorper<Frame::Grid>> excision_strahlkorpers =
      ylm::read_ylm_coefficients<Frame::Grid>(
          test_filename, target_names[0], num_requested_excision_strahlkorpers);

  const size_t num_requested_horizon_strahlkorpers = 2;
  const std::vector<Strahlkorper<Frame::Inertial>> horizon_strahlkorpers =
      ylm::read_ylm_coefficients<Frame::Inertial>(
          test_filename, target_names[1], num_requested_horizon_strahlkorpers);

  CHECK(excision_strahlkorpers.size() == num_requested_excision_strahlkorpers);
  CHECK(horizon_strahlkorpers.size() == num_requested_horizon_strahlkorpers);

  const ylm::Spherepack expected_excision_ylm{l_max, l_max};

  for (size_t i = 0; i < num_requested_excision_strahlkorpers; i++) {
    const auto& strahlkorper = excision_strahlkorpers[i];
    // const auto& ylm = strahlkorper.ylm_spherepack();

    CHECK(strahlkorper.l_max() == l_max);
    CHECK(strahlkorper.expansion_center() == expansion_center);
    CHECK(strahlkorper.coefficients().size() ==
          expected_excision_ylm.spectral_size());

    const DataVector ylm_coefficients = strahlkorper.coefficients();
    // std::cout << "ylm_coefficients :" << ylm_coefficients << std::endl;
    // std::cout << "excision_data[" << i << "] : " << excision_data[i] <<
    // std::endl;

    size_t h5_data_column = num_non_coef_columns;
    SpherepackIterator iter(l_max, l_max);
    for (size_t l = 0; l <= l_max; l++) {
      for (int m = -l; m <= static_cast<int>(l); m++) {
        iter.set(l, m);
        CHECK(ylm_coefficients[iter()] == excision_data[i][h5_data_column]);
        h5_data_column++;
      }
    }
  }

  //   Delete the temporary file created for this test
  //   file_system::rm(test_filename, true);
}
