// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <charm++.h>
#include <cstddef>
#include <exception>
#include <string>

#include "Parallel/Phase.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/System/Abort.hpp"

namespace Parallel {
[[noreturn]] void global_cache_pointer_null_error() {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  CkError(
      "Global cache pointer is null. This is an internal inconsistency "
      "error. Please file an issue.");
  sys::abort("");
}

[[noreturn]] void main_proxy_not_set_error() {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  CkError(
      "The main proxy has not been set in the global cache when terminating "
      "the component. This is an internal inconsistency error. Please file "
      "an issue.");
  sys::abort("");
}

std::string get_exception_message(const std::string& parallel_component_name,
                                  const std::string& array_index,
                                  const Phase phase,
                                  const size_t algorithm_step,
                                  const std::exception& exception) {
  return MakeString{} << "Component: " << parallel_component_name
                      << "\nArray Index: " << array_index << "\n"
                      << "Phase: " << phase << "\n"
                      << "Algorithm Step: " << algorithm_step << "\n"
                      << "Message: " << exception.what() << "\nType: "
                      << pretty_type::get_runtime_type_name(exception);
}
}  // namespace Parallel
