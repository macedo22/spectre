// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"

namespace ylm {
template <typename Frame>
std::vector<Strahlkorper<Frame>> read_ylm_coefficients(
    const std::string& file_name, const std::string& surface_name,
    const size_t requested_number_of_time_values);
}  // namespace ylm
