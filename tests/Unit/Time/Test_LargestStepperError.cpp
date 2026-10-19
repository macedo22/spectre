// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Time/LargestStepperError.hpp"
#include "Time/StepperErrorTolerances.hpp"

SPECTRE_TEST_CASE("Unit.Time.LargestStepperError", "[Unit][Time]") {
  CHECK(largest_stepper_error(20.0, 10.0, {.absolute = 2.0, .relative = 0.0}) ==
        5.0);
  CHECK(largest_stepper_error(20.0, 10.0, {.absolute = 0.0, .relative = 2.0}) ==
        1.0 / 6.0);
  CHECK(largest_stepper_error(0.0, 10.0, {.absolute = 0.0, .relative = 2.0}) ==
        0.5);

  CHECK(largest_stepper_error(20.0, 10.0, {.absolute = 2.0, .relative = 3.0}) ==
        10.0 / (2.0 + 3.0 * 30.0));

  CHECK(largest_stepper_error(std::complex<double>(12.0, 5.0),
                              std::complex<double>(-3.0, -4.0),
                              {.absolute = 2.0, .relative = 3.0}) ==
        approx(5.0 / (2.0 + 3.0 * 13.0)));

  {
    const DataVector values{10.0, 0.0};
    const DataVector errors{5.0, 2.0};
    const StepperErrorTolerances mostly_abs{.absolute = 2.0, .relative = 0.1};
    const StepperErrorTolerances mostly_rel{.absolute = 0.1, .relative = 2.0};
    CHECK(largest_stepper_error(values, errors, mostly_abs) ==
          largest_stepper_error(10.0, 5.0, mostly_abs));
    CHECK(largest_stepper_error(values, errors, mostly_rel) ==
          largest_stepper_error(0.0, 2.0, mostly_rel));

    const auto diagnostics =
        stepper_error_diagnostics(values, errors, mostly_rel);
    CHECK(diagnostics.number_of_values == 2);
    CHECK(diagnostics.maximum_absolute_error == 5.0);
    CHECK(diagnostics.maximum_absolute_error_index == 0);
    CHECK(diagnostics.maximum_absolute_step_start_value == 10.0);
    CHECK(diagnostics.maximum_absolute_step_start_value_index == 0);
    CHECK(diagnostics.maximum_pointwise_normalized_error == approx(2.0 / 4.1));
    CHECK(diagnostics.maximum_pointwise_normalized_error_index == 1);
    CHECK(diagnostics.absolute_error_at_maximum_pointwise_normalized_error ==
          2.0);
    CHECK(
        diagnostics
            .absolute_step_start_value_at_maximum_pointwise_normalized_error ==
        0.0);
    CHECK(
        diagnostics
            .absolute_step_end_estimate_at_maximum_pointwise_normalized_error ==
        2.0);
    CHECK(diagnostics.normalization_at_maximum_pointwise_normalized_error ==
          approx(4.1));

    auto diagnostics_with_step_end = diagnostics;
    record_step_end_value(diagnostics_with_step_end, DataVector{12.0, 30.0});
    CHECK(diagnostics_with_step_end.maximum_absolute_step_end_value == 30.0);
    CHECK(diagnostics_with_step_end.maximum_absolute_step_end_value_index == 1);
  }

  {
    const ComplexDataVector values{std::complex<double>{10.0, 3.0},
                                   std::complex<double>{0.0, 0.0}};
    const ComplexDataVector errors{std::complex<double>{5.0, 1.0},
                                   std::complex<double>{1.0, 3.0}};
    const StepperErrorTolerances mostly_abs{.absolute = 2.0, .relative = 0.1};
    const StepperErrorTolerances mostly_rel{.absolute = 0.1, .relative = 2.0};
    // Vectorized complex math may not be identical to std::complex operations.
    CHECK(largest_stepper_error(values, errors, mostly_abs) ==
          approx(largest_stepper_error(values[0], errors[0], mostly_abs)));
    CHECK(largest_stepper_error(values, errors, mostly_rel) ==
          approx(largest_stepper_error(values[1], errors[1], mostly_rel)));
  }

  {
    const DataVector values{};
    const DataVector errors{};
    const StepperErrorTolerances mostly_abs{.absolute = 2.0, .relative = 0.1};
    const StepperErrorTolerances mostly_rel{.absolute = 0.1, .relative = 2.0};
    CHECK(largest_stepper_error(values, errors, mostly_abs) == 0.0);
    CHECK(largest_stepper_error(values, errors, mostly_rel) == 0.0);
    auto diagnostics = stepper_error_diagnostics(values, errors, mostly_abs);
    CHECK(diagnostics.number_of_values == 0);
    CHECK(diagnostics.maximum_absolute_error_index ==
          StepperErrorDiagnostics::invalid_index);
    CHECK(diagnostics.maximum_absolute_step_start_value_index ==
          StepperErrorDiagnostics::invalid_index);
    CHECK(diagnostics.maximum_pointwise_normalized_error_index ==
          StepperErrorDiagnostics::invalid_index);
    record_step_end_value(diagnostics, DataVector{});
    CHECK(diagnostics.maximum_absolute_step_end_value_index ==
          StepperErrorDiagnostics::invalid_index);
  }
}
