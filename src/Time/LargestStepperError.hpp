// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>

#include "Time/StepperErrorEstimate.hpp"

/// \cond
class ComplexDataVector;
class DataVector;
class StepperErrorTolerances;
/// \endcond

/*!
 * \ingroup TimeGroup
 * \brief Calculate the pointwise worst error.
 *
 * \details For a `double`, or `std::complex<double>`, calculates
 *
 * \f{equation}
 *   \frac{e}{a + r \max(|v|, |v + e|)}
 * \f}
 *
 * where $v$ is \p values, $e$ is \p errors, and $a$ and $r$ are the
 * tolerances from \p tolerances.  For vector types, calculates the
 * largest error over all the points.  Only the `absolute` and
 * `relative` fields from \p tolerances are used.
 *
 * Instantiated for `double`, `std::complex<double>`, `DataVector`,
 * and `ComplexDataVector`.
 */
template <typename T>
double largest_stepper_error(const T& values, const T& errors,
                             const StepperErrorTolerances& tolerances);

/*!
 * \ingroup TimeGroup
 * \brief Collect maxima and locations useful for diagnosing the pointwise
 * error measure returned by `largest_stepper_error`.
 *
 * This function is observational only.  The time-step controller continues to
 * use the value returned by `largest_stepper_error`.
 */
template <typename T>
StepperErrorDiagnostics stepper_error_diagnostics(
    const T& step_start_values, const T& errors,
    const StepperErrorTolerances& tolerances);

/*!
 * \ingroup TimeGroup
 * \brief Add the maximum of the accepted step-end solution to diagnostics.
 */
template <typename T>
void record_step_end_value(StepperErrorDiagnostics& diagnostics,
                           const T& step_end_values);
