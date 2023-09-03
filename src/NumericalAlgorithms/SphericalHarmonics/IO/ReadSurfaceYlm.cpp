// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"

namespace ylm {
namespace {
// checks if a double is a nonnegative integer
bool is_nonnegative_int(const double n) {
  return n == abs(n) and n == floor(n);
}

template <typename Frame>
Strahlkorper<Frame> read_surface_ylm_row(const Matrix& ylm_data,
                                         const size_t row_number) {
  const size_t l_max_column_number = 4;
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
  // l_max == m_max
  const size_t spectral_size = Spherepack::spectral_size(l_max, l_max);
  // We only write and store half of the coefficients. This is the minimum
  // number of coefficients needed to describe the strahlkorper given the l_max,
  // i.e. columns of 0.0 for higher coefficients are okay
  const size_t min_expected_num_coefficients = spectral_size / 2;
  const size_t num_non_coef_headers = 5;
  const size_t min_expected_num_columns =
      min_expected_num_coefficients + num_non_coef_headers;
  const size_t actual_num_columns = ylm_data.columns();

  const std::string frame{get_output(Frame{})};
  const std::string expected_format{
      MakeString{}
      << "The expected format of the data is \'Time, " << frame
      << "ExpansionCenter_x, " << frame << "ExpansionCenter_y, " << frame
      << "Expansion_Center_z, Lmax, coef(0,0), coef(1,-1), coef(1,0), "
         "coef(1,1), coef(2,-2), coef(2,-1), coef(2,0), coef(2,1), coef(2,2), "
         "..., coef(Lmax,Lmax), [0.0...]\', where the number of coefficients "
         "is equal to (Lmax + 1)^2 and the coefficient columns may be padded "
         "with columns of 0.0 for any higher order coefficients for l > Lmax."};
  if (actual_num_columns < min_expected_num_columns) {
    ERROR("Row "
          << row_number
          << " of the Ylm data does not have the expected format. For Lmax = "
          << l_max << ", expected at least " << min_expected_num_columns
          << " columns.\n\n"
          << expected_format);
  }

  ModalVector spectral_coefficients(spectral_size, 0.0);
  size_t coef_column_number = num_non_coef_headers;
  SpherepackIterator iter(l_max, l_max);
  // read in expected coefficients for the given l_max
  for (size_t l = 0; l <= l_max; l++) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
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

  Strahlkorper<Frame> strahlkorper(l_max, l_max, spectral_coefficients,
                                   expansion_center);

  return strahlkorper;
}
}  // namespace

template <typename Frame>
std::vector<Strahlkorper<Frame>> read_surface_ylm(
    const std::string& file_name, const std::string& surface_subfile_name,
    const size_t requested_number_of_times_from_end) {
  h5::H5File<h5::AccessType::ReadOnly> file{file_name};
  const std::string ylm_subfile_name{std::string{"/"} + surface_subfile_name};
  const auto& ylm_file = file.get<h5::Dat>(ylm_subfile_name);
  const auto& ylm_legend = ylm_file.get_legend();

  const std::string expected_frame{get_output(Frame{})};
  const auto check_frame = [&expected_frame, &ylm_legend](const size_t column) {
    const std::string& expansion_center_header = gsl::at(ylm_legend, column);
    const size_t frame_name_length = expansion_center_header.find("Expansion");
    if (frame_name_length == 0 or frame_name_length == std::string::npos) {
      ERROR(
          "The frame type for an expansion center coordinate was not found in "
          "the Ylm subfile legend in column "
          << column
          << ". Expansion center column names are expected to have the format "
             "\'{Frame}ExpansionCenter_{x, y, or z}\'.");
    }
    const std::string read_frame =
        expansion_center_header.substr(0, frame_name_length);
    if (read_frame != expected_frame) {
      ERROR("The frame type in column "
            << column << " (" << read_frame
            << ") does not match the expected frame for the Strahlkorper to "
               "construct ("
            << expected_frame << ")");
    }
  };
  check_frame(1);  // for {Frame}ExpansionCenter_x
  check_frame(2);  // for {Frame}ExpansionCenter_y
  check_frame(3);  // for {Frame}ExpansionCenter_z

  // number of rows available
  const size_t total_number_of_times = gsl::at(ylm_file.get_dimensions(), 0);
  if (total_number_of_times == 0) {
    ERROR("The Ylm data to read from contain 0 rows (times) of data.");
  }

  ASSERT(requested_number_of_times_from_end > 0,
         "Must request to read in at least one row (time) of Ylm data.");

  if (requested_number_of_times_from_end > total_number_of_times) {
    ERROR("The requested number of time values ("
          << requested_number_of_times_from_end
          << ") is more than the number of rows in the Ylm data that was read "
             "in ("
          << total_number_of_times << ")\n");
  }

  std::vector<size_t> columns(ylm_legend.size());
  std::iota(std::begin(columns), std::end(columns), 0);
  // grab all columns of the last requested_number_of_times_from_end rows
  const auto ylm_data_subset = ylm_file.get_data_subset(
      columns, total_number_of_times - requested_number_of_times_from_end,
      requested_number_of_times_from_end);

  std::vector<Strahlkorper<Frame>> strahlkorpers(
      requested_number_of_times_from_end);
  for (size_t i = 0; i < requested_number_of_times_from_end; i++) {
    strahlkorpers[i] = read_surface_ylm_row<Frame>(ylm_data_subset, i);
  }

  file.close_current_object();
  return strahlkorpers;
}
}  // namespace ylm

#define FRAMETYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template std::vector<ylm::Strahlkorper<FRAMETYPE(data)>>         \
  ylm::read_surface_ylm<>(const std::string& file_name,            \
                          const std::string& surface_subfile_name, \
                          size_t requested_number_of_times_from_end);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Inertial, Frame::Distorted))

#undef INSTANTIATE
#undef FRAMETYPE
