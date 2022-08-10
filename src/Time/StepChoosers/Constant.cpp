// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/Constant.hpp"

#include <sstream>

#include "Options/ParseOptions.hpp"

namespace StepChoosers {
namespace Constant_detail {
// This function lets us avoid including ParseOptions.hpp in the
// header.
double parse_options(const Options::Option& options) {
  const auto value = options.parse_as<double>();
  if (value <= 0.) {
    std::stringstream ss;
    ss << "Requested step magnitude should be positive.";
    PARSE_ERROR(options.context(), ss);
  }
  return value;
}
}  // namespace Constant_detail
}  // namespace StepChoosers
