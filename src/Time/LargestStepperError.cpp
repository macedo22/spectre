// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/LargestStepperError.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <typename T>
double largest_stepper_error(const T& values, const T& errors,
                             const StepperErrorTolerances& tolerances) {
  using std::abs;
  // Outer call to max() may be from blaze or an identity function
  // from ContainerHelpers.hpp for doubles.
  using ::max;
  // Inner max() is either blaze or std::max.
  using std::max;
  return max(abs(errors) /
             (tolerances.absolute +
              tolerances.relative * max(abs(values), abs(values + errors))));
}

template <typename T>
StepperErrorDiagnostics stepper_error_diagnostics(
    const T& step_start_values, const T& errors,
    const StepperErrorTolerances& tolerances) {
  const size_t number_of_values = get_size(step_start_values);
  ASSERT(number_of_values == get_size(errors),
         "Cannot compare " << number_of_values << " solution values with "
                           << get_size(errors) << " error values.");

  StepperErrorDiagnostics result{};
  result.number_of_values = number_of_values;
  using std::abs;
  for (size_t i = 0; i < number_of_values; ++i) {
    const double absolute_step_start_value =
        abs(get_element(step_start_values, i));
    const double absolute_error = abs(get_element(errors, i));
    const double absolute_step_end_estimate =
        abs(get_element(step_start_values, i) + get_element(errors, i));
    const double normalization =
        tolerances.absolute +
        tolerances.relative *
            std::max(absolute_step_start_value, absolute_step_end_estimate);
    const double pointwise_normalized_error = absolute_error / normalization;

    if (result.maximum_absolute_error_index ==
            StepperErrorDiagnostics::invalid_index or
        absolute_error > result.maximum_absolute_error) {
      result.maximum_absolute_error = absolute_error;
      result.maximum_absolute_error_index = i;
    }
    if (result.maximum_absolute_step_start_value_index ==
            StepperErrorDiagnostics::invalid_index or
        absolute_step_start_value > result.maximum_absolute_step_start_value) {
      result.maximum_absolute_step_start_value = absolute_step_start_value;
      result.maximum_absolute_step_start_value_index = i;
    }
    if (result.maximum_pointwise_normalized_error_index ==
            StepperErrorDiagnostics::invalid_index or
        pointwise_normalized_error >
            result.maximum_pointwise_normalized_error) {
      result.maximum_pointwise_normalized_error = pointwise_normalized_error;
      result.maximum_pointwise_normalized_error_index = i;
      result.absolute_error_at_maximum_pointwise_normalized_error =
          absolute_error;
      result.absolute_step_start_value_at_maximum_pointwise_normalized_error =
          absolute_step_start_value;
      result.absolute_step_end_estimate_at_maximum_pointwise_normalized_error =
          absolute_step_end_estimate;
      result.normalization_at_maximum_pointwise_normalized_error =
          normalization;
    }
  }
  return result;
}

template <typename T>
void record_step_end_value(StepperErrorDiagnostics& diagnostics,
                           const T& step_end_values) {
  ASSERT(diagnostics.number_of_values == get_size(step_end_values),
         "Cannot combine diagnostics for " << diagnostics.number_of_values
                                           << " values with a step end holding "
                                           << get_size(step_end_values)
                                           << " values.");
  using std::abs;
  for (size_t i = 0; i < diagnostics.number_of_values; ++i) {
    const double absolute_step_end_value = abs(get_element(step_end_values, i));
    if (diagnostics.maximum_absolute_step_end_value_index ==
            StepperErrorDiagnostics::invalid_index or
        absolute_step_end_value > diagnostics.maximum_absolute_step_end_value) {
      diagnostics.maximum_absolute_step_end_value = absolute_step_end_value;
      diagnostics.maximum_absolute_step_end_value_index = i;
    }
  }
}

#define TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template double largest_stepper_error(                                    \
      const TYPE(data) & values, const TYPE(data) & errors,                 \
      const StepperErrorTolerances& tolerances);                            \
  template StepperErrorDiagnostics stepper_error_diagnostics(               \
      const TYPE(data) & step_start_values, const TYPE(data) & errors,      \
      const StepperErrorTolerances& tolerances);                            \
  template void record_step_end_value(StepperErrorDiagnostics& diagnostics, \
                                      const TYPE(data) & step_end_values);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, std::complex<double>, DataVector,
                                      ComplexDataVector))

#undef INSTANTIATE
#undef TYPE
