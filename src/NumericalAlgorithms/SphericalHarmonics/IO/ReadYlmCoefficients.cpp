// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadYlmCoefficients.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ylm {
namespace {
template <typename Frame>
Strahlkorper<Frame> read_ylm_coefficients_row(const Matrix& ylm_data,
                                              const size_t row_number) {
  const std::array<double, 3> expansion_center{{ylm_data(row_number, 1),
                                                ylm_data(row_number, 2),
                                                ylm_data(row_number, 3)}};
  const double l_max = ylm_data(row_number, 4);
  // number of terms in
  // \sum_{l=0}^{l_{max}} \sum_{m=-l}^{l} F^{lm} Y^{lm}(\theta,\phi) is the
  // sum of the first (l_max + 1) odd numbers, which is (l_max + 1)^2
  // const size_t expected_num_cofficients = square(l_max + 1);
  // l_max == m_max
  const size_t spectral_size = Spherepack::spectral_size(l_max, l_max);
  DataVector spectral_coefficients(spectral_size, 0.0);

  // std::vector<double> ylm_coefficients;
  // ylm_coefficients.reserve(expected_num_cofficients);
  // 5 = column # of first coefficient
  size_t coef_column_number = 5;
  SpherepackIterator iter(l_max, l_max);
    for (size_t l = 0; l <= l_max; l++) {
      for (int m = -l; m <= static_cast<int>(l); m++) {
        iter.set(l, m);
        // 5 = number of non-coef columns preceding coef columns
        const double coefficient = ylm_data(row_number, coef_column_number);
        // ylm_coefficients.emplace_back(coefficient);
        spectral_coefficients[iter()] = coefficient;
        coef_column_number++;
      }
    }

    Strahlkorper<Frame> strahlkorper(
        l_max, l_max, spectral_coefficients, expansion_center,
        StrahlkorperContructorData::SpectralCoefficients);

    return strahlkorper;
}
}  // namespace

template <typename Frame>
std::vector<Strahlkorper<Frame>> read_ylm_coefficients(
    const std::string& file_name, const std::string& surface_name,
    const size_t requested_number_of_time_values) {
  h5::H5File<h5::AccessType::ReadOnly> file{file_name};
  const std::string ylm_subfile_name{std::string{"/"} + surface_name + "_Ylm"};
  const auto& ylm_file = file.get<h5::Dat>(ylm_subfile_name);
  const auto& ylm_data = ylm_file.get_data();

  const size_t total_number_of_time_values = ylm_data.rows();
  if (total_number_of_time_values == 0) {
    ERROR("The input Ylm data contains no data.");
  }

  if (requested_number_of_time_values == 0) {
    ERROR("No Ylm data is being requested to be read in.");
  }

  if (requested_number_of_time_values > total_number_of_time_values) {
    ERROR("The requested number of time values ("
          << requested_number_of_time_values
          << ") is more than the number of rows in the input Ylm data ("
          << total_number_of_time_values << ")\n");
  }

  const size_t l_max_column_number = 4;
  // TODO : assert it's a positive integer?
  const size_t first_row_l_max = ylm_data(0, l_max_column_number);
  // number of terms in
  // \sum_{l=0}^{l_{max}} \sum_{m=-l}^{l} F^{lm} Y^{lm}(\theta,\phi) is the
  // sum of the first (l_max + 1) odd numbers, which is (l_max + 1)^2
  const size_t expected_num_coefficients = square(first_row_l_max + 1);
  // 5 columns with non-coef data: time, 3 coords of center, and Lmax
  const size_t expected_num_columns = expected_num_coefficients + 5;
  const size_t actual_num_columns = ylm_data.columns();

  // TODO : this error checking can maybe be moved into the row function
  const std::string expected_ylm_legend =
      "\'Time, ExpansionCenter_x, ExpansionCenter_y, Expansion_Center_z, "
      "Lmax, coefs...\'";
  const std::string format_error_message{
      "The Ylm data does not have the expected format. "
      "The expected format of the data is " +
      expected_ylm_legend +
      ", where the number of coefficients (coefs...) is equal to "
      "(Lmax + 1)^2"};

  if (expected_num_columns != actual_num_columns) {
    ERROR(format_error_message);
  }
  for (size_t i = 1; i < requested_number_of_time_values; i++) {
    const size_t l_max = ylm_data(i, l_max_column_number);
    if (l_max != first_row_l_max) {
      ERROR(format_error_message);
    }
  }

  std::vector<Strahlkorper<Frame>> strahlkorpers(
      requested_number_of_time_values);
  for (size_t i = 0, row_number = total_number_of_time_values -
                                  requested_number_of_time_values;
       i < requested_number_of_time_values; i++, row_number++) {
    strahlkorpers[i] = read_ylm_coefficients_row<Frame>(ylm_data, row_number);
  }

  file.close_current_object();
  return strahlkorpers;
}
}  // namespace ylm

#define FRAMETYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                    \
  template std::vector<Strahlkorper<FRAMETYPE(data)>>           \
  ylm::read_ylm_coefficients<>(const std::string& file_name,    \
                               const std::string& surface_name, \
                               const size_t requested_number_of_time_values);

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial))

#undef INSTANTIATE
#undef FRAMETYPE
