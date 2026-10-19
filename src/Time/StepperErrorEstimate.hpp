// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "Time/Time.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup TimeGroup
/// Detailed diagnostic information about a time-stepper error estimate.
///
/// The indices refer to the flattened storage passed to the time stepper.  For
/// a `Variables`, the grid-point index varies fastest, followed by the
/// independent tensor-component index.
struct StepperErrorDiagnostics {
  static constexpr size_t invalid_index = std::numeric_limits<size_t>::max();

  size_t number_of_values = 0;

  double maximum_absolute_error = 0.0;
  size_t maximum_absolute_error_index = invalid_index;

  double maximum_absolute_step_start_value = 0.0;
  size_t maximum_absolute_step_start_value_index = invalid_index;

  double maximum_absolute_step_end_value = 0.0;
  size_t maximum_absolute_step_end_value_index = invalid_index;

  double maximum_pointwise_normalized_error = 0.0;
  size_t maximum_pointwise_normalized_error_index = invalid_index;
  double absolute_error_at_maximum_pointwise_normalized_error = 0.0;
  double absolute_step_start_value_at_maximum_pointwise_normalized_error = 0.0;
  double absolute_step_end_estimate_at_maximum_pointwise_normalized_error = 0.0;
  double normalization_at_maximum_pointwise_normalized_error = 0.0;

  void pup(PUP::er& p);
};

/// \ingroup TimeGroup
/// Estimate of the TimeStepper truncation error.
struct StepperErrorEstimate {
  /// Start of the step the estimate is for.
  Time step_time{};
  /// Size of the step the estimate is for.
  TimeDelta step_size{};
  /// Order of accuracy of the primary estimate.
  size_t order{};
  /// Error estimates.  The entry `errors[order]`, with `order` the
  /// previous member of this struct, is the estimated error of the
  /// time step, and has a convenience accessor `step_error()`.  Other
  /// entries are estimations of the error if the order of a
  /// variable-order stepper were to be changed.  The error
  /// `errors[i]` should scale approximately as $(\Delta t)^{i + 1}$.
  std::array<std::optional<double>, 8> errors{};

  /// Additional data collected only when explicitly requested for debugging.
  std::optional<StepperErrorDiagnostics> diagnostics{};

  StepperErrorEstimate() = default;
  StepperErrorEstimate(Time step_time_in, TimeDelta step_size_in,
                       size_t order_in, double step_error_in);

  /// Convenience accessor for `errors[order].value()`.
  double step_error() const;

  void pup(PUP::er& p);
};
