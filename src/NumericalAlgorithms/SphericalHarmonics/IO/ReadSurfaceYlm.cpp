// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"

#include <array>
#include <cmath>
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
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ylm {
namespace {
const size_t l_max_column_number = 4;
const size_t num_non_coef_headers = 5;

// checks if a double is a nonnegative integer
bool is_nonnegative_int(const double n) {
  return n == abs(n) and n == floor(n);
}

template <typename Frame>
Strahlkorper<Frame> read_surface_ylm_row(const Matrix& ylm_data,
                                         const size_t row_number) {
  const std::array<double, 3> expansion_center{{ylm_data(row_number, 1),
                                                ylm_data(row_number, 2),
                                                ylm_data(row_number, 3)}};
  const double l_max_from_file = ylm_data(row_number, l_max_column_number);
  if (not is_nonnegative_int(l_max_from_file)) {
    ERROR("Row " << row_number << " of the Ylm data has an invalid Lmax value ("
                 << l_max_from_file << ") in column " << l_max_column_number
                 << ". The value of Lmax should be a nonnegative integer.");
  }

  const size_t l_max = ylm_data(row_number, l_max_column_number);
  // number of terms in
  // \sum_{l=0}^{l_{max}} \sum_{m=-l}^{l} F^{lm} Y^{lm}(\theta,\phi) is the
  // sum of the first (l_max + 1) odd numbers, which is (l_max + 1)^2. This is
  // the minimum number of coefficients needed to describe the strahlkorper
  // given the l_max, i.e. columns of 0.0 for higher coefficients are okay
  const size_t min_expected_num_coefficients = square(l_max + 1);
  const size_t min_expected_num_columns =
      min_expected_num_coefficients + num_non_coef_headers;
  const size_t actual_num_columns = ylm_data.columns();
  const std::string expected_format{
      "The expected format of the data is \'Time, ExpansionCenter_x, "
      "ExpansionCenter_y, Expansion_Center_z, Lmax, Re(0,0), Im(1,-1), "
      "Re(1,0), Re(1,1), Im(2,-2), Im(2,-1), Re(2,0), Re(2,1), Re(2,2), ..., "
      "Re(Lmax,Lmax), [0.0...]\', where the number of coefficients is equal to "
      "(Lmax + 1)^2 and the coefficient columns may be padded with columns of "
      "0.0 for any higher order coefficients for l > Lmax."};
  if (actual_num_columns < min_expected_num_columns) {
    ERROR("Row "
          << row_number
          << " of the Ylm data does not have the expected format. For Lmax = "
          << l_max << ", expected at least " << min_expected_num_columns
          << " columns.\n\n"
          << expected_format);
  }

  // l_max == m_max
  const size_t spectral_size = Spherepack::spectral_size(l_max, l_max);
  DataVector spectral_coefficients(spectral_size, 0.0);

  size_t coef_column_number = num_non_coef_headers;
  SpherepackIterator iter(l_max, l_max);
  // read in expected coefficients for the given l_max
  for (size_t l = 0; l <= l_max; l++) {
    for (int m = -l; m <= static_cast<int>(l); m++) {
      iter.set(l, m);

      const double coefficient = ylm_data(row_number, coef_column_number);
      spectral_coefficients[iter()] = coefficient;

      coef_column_number++;
    }
  }
  // make sure any higher order coefficients (l > l_max) present in the data
  // that was read in are 0.0
  while (coef_column_number < actual_num_columns) {
    if (ylm_data(row_number, coef_column_number) != 0.0) {
      ERROR("Row " << row_number << " of the Ylm data has Lmax " << l_max
                   << " but non-zero coefficients for l > Lmax.\n\n"
                   << expected_format);
    }
    coef_column_number++;
  }

  Strahlkorper<Frame> strahlkorper(
      l_max, l_max, spectral_coefficients, expansion_center,
      StrahlkorperContructorData::SpectralCoefficients);

  return strahlkorper;
}
}  // namespace

template <typename Frame>
std::vector<Strahlkorper<Frame>> read_surface_ylm(
    const std::string& file_name, const std::string& surface_subfile_name,
    const size_t requested_number_of_time_values) {
  h5::H5File<h5::AccessType::ReadOnly> file{file_name};
  const std::string ylm_subfile_name{std::string{"/"} + surface_subfile_name};
  const auto& ylm_file = file.get<h5::Dat>(ylm_subfile_name);
  const auto& ylm_data = ylm_file.get_data();

  const size_t total_number_of_time_values = ylm_data.rows();
  if (total_number_of_time_values == 0) {
    ERROR("The Ylm data to read from contains 0 rows (times) of data.");
  }

  ASSERT(requested_number_of_time_values > 0,
         "Must request to read in at least one row (time) of Ylm data.");

  if (requested_number_of_time_values > total_number_of_time_values) {
    ERROR("The requested number of time values ("
          << requested_number_of_time_values
          << ") is more than the number of rows in the Ylm data that was read "
             "in ("
          << total_number_of_time_values << ")\n");
  }

  std::vector<Strahlkorper<Frame>> strahlkorpers(
      requested_number_of_time_values);
  for (size_t i = 0, row_number = total_number_of_time_values -
                                  requested_number_of_time_values;
       i < requested_number_of_time_values; i++, row_number++) {
    strahlkorpers[i] = read_surface_ylm_row<Frame>(ylm_data, row_number);
  }

  file.close_current_object();
  return strahlkorpers;
}
}  // namespace ylm

#define FRAMETYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::vector<Strahlkorper<FRAMETYPE(data)>> ylm::read_surface_ylm<>( \
      const std::string& file_name, const std::string& surface_subfile_name,   \
      const size_t requested_number_of_time_values);

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial))

#undef INSTANTIATE
#undef FRAMETYPE
