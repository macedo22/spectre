// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/CollectStepperErrorTolerances.hpp"

#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace collect_stepper_error_tolerances_detail {
void process_request(
    const gsl::not_null<
        std::unordered_map<std::type_index, StepperErrorTolerances>*>
        tolerances,
    const RequestsStepperErrorTolerances& tolerance_request) {
  auto new_requests = tolerance_request.tolerances();
  while (not new_requests.empty()) {
    auto request = new_requests.extract(new_requests.begin());
    if (request.mapped().estimates != StepperErrorTolerances::Estimates::None) {
      const auto inserted = tolerances->insert(std::move(request));
      if (not inserted.inserted) {
        const auto& new_tolerances = inserted.node.mapped();
        auto& old_tolerances = inserted.position->second;
        const bool request_diagnostics =
            new_tolerances.diagnostics or old_tolerances.diagnostics;
        auto new_tolerances_without_diagnostics = new_tolerances;
        auto old_tolerances_without_diagnostics = old_tolerances;
        new_tolerances_without_diagnostics.diagnostics = false;
        old_tolerances_without_diagnostics.diagnostics = false;
        if (new_tolerances_without_diagnostics !=
            old_tolerances_without_diagnostics) {
          ERROR_NO_TRACE(
              "All time stepping error tolerances for one set of variables "
              "must be the same, but found differing values:\n"
              << new_tolerances << "\n"
              << old_tolerances);
        }
        old_tolerances.diagnostics = request_diagnostics;
      }
    }
  }
}
}  // namespace collect_stepper_error_tolerances_detail
