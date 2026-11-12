// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/History.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Tags {
/// Normalized embedded error maximized over components separately at each
/// point.
struct PointwiseNormalizedTimeStepperError : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Reconstruct the completed DP5 embedded error only when observed. DP5 retains
/// the required derivatives and step-start value through history cleanup at
/// the end of the step. Observe at step/slab boundaries, before the next step's
/// RecordTimeStepperData and history cleanup replace that history.
template <size_t Dim, typename VariablesTag>
struct PointwiseNormalizedTimeStepperErrorCompute
    : PointwiseNormalizedTimeStepperError,
      db::ComputeTag {
  using base = PointwiseNormalizedTimeStepperError;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, ::Tags::StepperErrors<VariablesTag>,
                 ::Tags::HistoryEvolvedVariables<VariablesTag>,
                 ::Tags::StepperErrorTolerances<VariablesTag>,
                 ::Tags::TimeStepper<::TimeStepper>, ::Tags::TimeStepId>;

  static void function(
      const gsl::not_null<return_type*> result, const Mesh<Dim>& mesh,
      const typename ::Tags::StepperErrors<VariablesTag>::type& errors,
      const TimeSteppers::History<typename VariablesTag::type>& history,
      const StepperErrorTolerances& tolerances, const ::TimeStepper& stepper,
      const TimeStepId& id) {
    const auto* const dp5 =
        dynamic_cast<const TimeSteppers::DormandPrince5*>(&stepper);
    if (dp5 == nullptr) {
      ERROR_NO_TRACE(
          "PointwiseNormalizedTimeStepperError currently requires "
          "DormandPrince5.");
    }
    const size_t npoints = mesh.number_of_grid_points();
    get(*result) =
        DataVector(npoints, std::numeric_limits<double>::quiet_NaN());
    if (not errors[1].has_value()) {
      return;
    }
    const auto& estimate = *errors[1];
    if (id.substep() != 0 or
        id.step_time() != estimate.step_time + estimate.step_size or
        history.size() != 1 or history.substeps().size() != 6 or
        history.back().time_step_id.step_time() != estimate.step_time or
        not history.back().value.has_value()) {
      ERROR_NO_TRACE(
          "PointwiseNormalizedTimeStepperError needs the completed DP5 "
          "history. Observe it in EventsAndTriggersAtSteps or "
          "EventsAndTriggersAtSlabs, before changing the mesh; do not use "
          "dense-output observations.");
    }
    const auto& start = *history.back().value;
    if (start.number_of_grid_points() != npoints or
        history.back().derivative.number_of_grid_points() != npoints) {
      ERROR_NO_TRACE(
          "PointwiseNormalizedTimeStepperError history and "
          "observation mesh differ.");
    }
    for (size_t stage = 0; stage < 6; ++stage) {
      if (history.substeps()[stage].derivative.number_of_grid_points() !=
              npoints or
          history.substeps()[stage].time_step_id.step_time() !=
              estimate.step_time) {
        ERROR_NO_TRACE(
            "PointwiseNormalizedTimeStepperError has inconsistent "
            "stage history.");
      }
    }
    const auto& tableau = dp5->butcher_tableau();
    const auto weight = [&tableau](const size_t stage) {
      return (stage < tableau.result_coefficients.size()
                  ? tableau.result_coefficients[stage]
                  : 0.0) -
             tableau.error_coefficients[stage];
    };
    const double dt = estimate.step_size.value();
    get(*result) = 0.0;
    // Variables storage has point index varying fastest. Apply the same
    // arithmetic ordering and normalization as RungeKutta::step_error and
    // largest_stepper_error, reducing only over components, not points.
    for (size_t flat = 0; flat < start.size(); ++flat) {
      double delta = weight(0) * dt * history.back().derivative.data()[flat];
      for (size_t stage = 1; stage < 7; ++stage) {
        if (weight(stage) != 0.0) {
          delta += weight(stage) * dt *
                   history.substeps()[stage - 1].derivative.data()[flat];
        }
      }
      const double value = start.data()[flat];
      const double error =
          std::abs(delta) /
          (tolerances.absolute +
           tolerances.relative *
               std::max(std::abs(value), std::abs(value + delta)));
      auto& point_error = get(*result)[flat % npoints];
      point_error = std::isnan(error) ? error : std::max(point_error, error);
    }
  }
};

/// The latest element-wide normalized stepper error, repeated at every point.
/// This is the stored error-control value, not a pointwise error field.
struct NormalizedTimeStepperError : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Start time of the step associated with NormalizedTimeStepperError.
struct TimeStepperErrorStepStart : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Size of the step associated with NormalizedTimeStepperError.
struct TimeStepperErrorStepSize : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Expose the latest stored error estimate on the element's full volume mesh.
/// Returns NaN when no estimate is available (e.g. at initialization).
template <size_t Dim, typename VariablesTag>
struct NormalizedTimeStepperErrorCompute : NormalizedTimeStepperError,
                                           db::ComputeTag {
  using base = NormalizedTimeStepperError;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, ::Tags::StepperErrors<VariablesTag>>;

  static void function(
      const gsl::not_null<return_type*> result, const Mesh<Dim>& mesh,
      const typename ::Tags::StepperErrors<VariablesTag>::type& errors) {
    get(*result) = DataVector(mesh.number_of_grid_points(),
                              errors[1].has_value()
                                  ? errors[1]->step_error()
                                  : std::numeric_limits<double>::quiet_NaN());
  }
};

template <size_t Dim, typename VariablesTag>
struct TimeStepperErrorStepStartCompute : TimeStepperErrorStepStart,
                                          db::ComputeTag {
  using base = TimeStepperErrorStepStart;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, ::Tags::StepperErrors<VariablesTag>>;

  static void function(
      const gsl::not_null<return_type*> result, const Mesh<Dim>& mesh,
      const typename ::Tags::StepperErrors<VariablesTag>::type& errors) {
    get(*result) = DataVector(mesh.number_of_grid_points(),
                              errors[1].has_value()
                                  ? errors[1]->step_time.value()
                                  : std::numeric_limits<double>::quiet_NaN());
  }
};

template <size_t Dim, typename VariablesTag>
struct TimeStepperErrorStepSizeCompute : TimeStepperErrorStepSize,
                                         db::ComputeTag {
  using base = TimeStepperErrorStepSize;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, ::Tags::StepperErrors<VariablesTag>>;

  static void function(
      const gsl::not_null<return_type*> result, const Mesh<Dim>& mesh,
      const typename ::Tags::StepperErrors<VariablesTag>::type& errors) {
    get(*result) = DataVector(mesh.number_of_grid_points(),
                              errors[1].has_value()
                                  ? errors[1]->step_size.value()
                                  : std::numeric_limits<double>::quiet_NaN());
  }
};
}  // namespace evolution::dg::Tags
