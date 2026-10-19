// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepperErrorEstimate.hpp"

#include <pup.h>
#include <pup_stl.h>

#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

void StepperErrorDiagnostics::pup(PUP::er& p) {
  p | number_of_values;
  p | maximum_absolute_error;
  p | maximum_absolute_error_index;
  p | maximum_absolute_step_start_value;
  p | maximum_absolute_step_start_value_index;
  p | maximum_absolute_step_end_value;
  p | maximum_absolute_step_end_value_index;
  p | maximum_pointwise_normalized_error;
  p | maximum_pointwise_normalized_error_index;
  p | absolute_error_at_maximum_pointwise_normalized_error;
  p | absolute_step_start_value_at_maximum_pointwise_normalized_error;
  p | absolute_step_end_estimate_at_maximum_pointwise_normalized_error;
  p | normalization_at_maximum_pointwise_normalized_error;
}

StepperErrorEstimate::StepperErrorEstimate(const Time step_time_in,
                                           const TimeDelta step_size_in,
                                           const size_t order_in,
                                           const double step_error_in)
    : step_time(step_time_in), step_size(step_size_in), order(order_in) {
  gsl::at(errors, order).emplace(step_error_in);
}

double StepperErrorEstimate::step_error() const {
  return gsl::at(errors, order).value();
}

void StepperErrorEstimate::pup(PUP::er& p) {
  p | step_time;
  p | step_size;
  p | order;
  p | errors;
  p | diagnostics;
}
