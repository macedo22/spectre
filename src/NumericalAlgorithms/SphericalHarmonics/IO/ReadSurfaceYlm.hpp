// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"

namespace ylm {
/// \ingroup SurfacesGroup
/// \brief Returns a list of `::Strahlkorper`s constructed from reading in
/// spherical harmonic data at a list of times
///
/// \details The `::Strahlkorper`s are constructed using data that is expected
/// to be in the format described by `intrp::callbacks::ObserveSurfaceData`.
///
/// \param file_name name of the h5 file containing the spherical harmonic data
/// \param surface_subfile_name name of the subfile within `file_name` that
/// contains the spherical harmonic data to read in
/// \param requested_number_of_time_values the number of times to read in
/// starting from the final time found in the written data
template <typename Frame>
std::vector<Strahlkorper<Frame>> read_surface_ylm(
    const std::string& file_name, const std::string& surface_subfile_name,
    const size_t requested_number_of_time_values);
}  // namespace ylm
