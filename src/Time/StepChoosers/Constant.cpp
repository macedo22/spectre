// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/Constant.hpp"

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/ParseOptions.tpp"

namespace StepChoosers::Constant_detail {
// This function lets us avoid including ParseOptions.hpp in the
// header.
double parse_options(const Options::Option& options) {
  const auto value = options.parse_as<double>();
  if (value <= 0.) {
    PARSE_ERROR(options.context(),
                "Requested step magnitude should be positive.");
  }
  return value;
}
}  // namespace StepChoosers::Constant_detail
