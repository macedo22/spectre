// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "DataStructures/DataBox/TagName.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Options/String.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
struct NoSuchType;
/// \endcond

namespace StepChoosers {
namespace ErrorControl_detail {
template <typename System, typename = std::void_t<>>
struct has_volume_dim : std::false_type {};

template <typename System>
struct has_volume_dim<System, std::void_t<decltype(System::volume_dim)>>
    : std::true_type {};

template <typename System, typename ErrorTags,
          bool HasVolumeDim = has_volume_dim<System>::value>
struct argument_tags {
  using type = ErrorTags;
};

template <typename System, typename... ErrorTags>
struct argument_tags<System, tmpl::list<ErrorTags...>, true> {
  using type =
      tmpl::list<ErrorTags..., domain::Tags::Element<System::volume_dim>,
                 ::Tags::TimeStepId>;
};

template <typename StepChooserUse>
std::optional<double> goal_from_variable(
    const std::array<std::optional<StepperErrorEstimate>, 2>& errors,
    double min_factor, double max_factor, double safety_factor);

struct ComponentLocation {
  std::string name{"unavailable"};
  size_t independent_component = StepperErrorDiagnostics::invalid_index;
  size_t grid_point = StepperErrorDiagnostics::invalid_index;
  size_t number_of_grid_points = 0;
};

template <typename VariablesTag>
ComponentLocation component_location(const size_t flat_index,
                                     const size_t number_of_values) {
  using VariablesType = typename VariablesTag::type;
  constexpr size_t number_of_components =
      VariablesType::number_of_independent_components;
  static_assert(number_of_components > 0);
  ComponentLocation result{};
  if (flat_index == StepperErrorDiagnostics::invalid_index or
      number_of_values == 0 or number_of_values % number_of_components != 0) {
    return result;
  }
  result.number_of_grid_points = number_of_values / number_of_components;
  result.independent_component = flat_index / result.number_of_grid_points;
  result.grid_point = flat_index % result.number_of_grid_points;
  if (result.independent_component >= number_of_components) {
    return result;
  }

  size_t first_component = 0;
  tmpl::for_each<typename VariablesType::tags_list>([&result, &first_component](
                                                        auto tag_v) {
    using Tag = tmpl::type_from<decltype(tag_v)>;
    using TensorType = typename Tag::type;
    if (result.name == "unavailable" and
        result.independent_component >= first_component and
        result.independent_component < first_component + TensorType::size()) {
      result.name = db::tag_name<Tag>() +
                    TensorType::component_suffix(result.independent_component -
                                                 first_component);
    }
    first_component += TensorType::size();
  });
  return result;
}
}  // namespace ErrorControl_detail

/*!
 * \brief Sets a goal based on time-stepper truncation error.
 *
 * \details The suggested step is calculated via a simple specialization of the
 * scheme suggested in \cite Hairer1993. We first compute the aggregated error
 * measure from the stepper error:
 *
 * \f[
 * E = \max_i(|E_i| / sc_i),
 * \f]
 *
 * where \f$E_i\f$ is the ODE error reported for each individual grid point,
 * reported by the time stepper, and \f$sc_i\f$ is the step control measure
 * determined by the tolerances:
 *
 * \f[
 * sc_i = Atol_i + \max(|y_i|,|y_i + E_i|) Rtol_i,
 * \f]
 *
 * and \f$y_i\f$ is the value of the function at the previous step at
 * grid point \f$i\f$.  (The estimate is more commonly done comparing
 * with the current step, but using the previous step avoids a memory
 * allocation in the TimeStepper and should not have a major effect on
 * the result.)
 *
 * When choosing a step size for LTS or when no record of previous
 * error is available, the step has size:
 *
 * \f[
 * h_{\text{new}} = h \cdot \min\left(F_{\text{max}},
 * \max\left(F_{\text{min}},
 * \frac{F_{\text{safety}}}{E^{1/(q + 1)}}\right)\right),
 * \f]
 *
 * where \f$h_{\text{new}}\f$ is the new suggested step size \f$h\f$ is the
 * previous step size, \f$F_{\text{max}}\f$ is the maximum factor by which we
 * allow the step to increase, \f$F_{\text{min}}\f$ is the minimum factor by
 * which we allow the step to decrease. \f$F_{\text{safety}}\f$ is the safety
 * factor on the computed error -- this forces the step size slightly lower
 * than we would naively compute so that the result of the step will likely be
 * within the target error. \f$q\f$ is the order of the stepper error
 * calculation. Intuitively, we should change the step less drastically for a
 * higher order stepper.
 *
 * When controlling slab size, after the first error calculation, the
 * error \f$E\f$ is recorded in the \ref DataBoxGroup "DataBox", and
 * subsequent error calculations use a simple PI scheme suggested in
 * \cite NumericalRecipes section 17.2.1:
 *
 * \f[
 * h_{\text{new}} = h \cdot \min\left(F_{\text{max}},
 * \max\left(F_{\text{min}},
 * F_{\text{safety}} E^{-0.7 / (q + 1)}
 * E_{\text{prev}}^{0.4 / (q + 1)}\right)\right),
 * \f]
 *
 * where \f$E_{\text{prev}}\f$ is the error computed in the previous
 * step.  This method is never used for choosing an LTS step because
 * the restriction of step size changes to factors of two was found to
 * interfere with the more gradual increase chosen by the PI
 * controller.
 */
template <typename StepChooserUse, typename System,
          typename = tmpl::conditional_t<
              tt::is_a_v<tmpl::list, typename System::variables_tag>,
              typename System::variables_tag,
              tmpl::list<typename System::variables_tag>>>
class ErrorControl;

template <typename StepChooserUse, typename System, typename... VariablesTags>
class ErrorControl<StepChooserUse, System, tmpl::list<VariablesTags...>>
    : public StepChooser<StepChooserUse>,
      public RequestsStepperErrorTolerances {
 public:
  /// \cond
  ErrorControl() = default;
  explicit ErrorControl(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ErrorControl);  // NOLINT
  /// \endcond

  struct AbsoluteTolerance {
    using type = double;
    static constexpr Options::String help{"Target absolute tolerance"};
    static type lower_bound() { return 0.0; }
  };

  struct RelativeTolerance {
    using type = double;
    static constexpr Options::String help{"Target relative tolerance"};
    static type lower_bound() { return 0.0; }
  };

  struct MaxFactor {
    using type = double;
    static constexpr Options::String help{
        "Maximum factor to increase the step by"};
    static type lower_bound() { return 1.0; }
  };

  struct MinFactor {
    using type = double;
    static constexpr Options::String help{
        "Minimum factor to increase the step by"};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };

  struct SafetyFactor {
    using type = double;
    static constexpr Options::String help{
        "Extra factor to apply to step estimate; can be used to decrease step "
        "size to improve step acceptance rate."};
    static type lower_bound() { return 0.0; }
  };

  struct Diagnostics {
    using type = bool;
    static constexpr Options::String help{
        "Write per-element error-control diagnostics to "
        "'TimeStepperErrorDiagnostics.txt'. This is intended for debugging "
        "and can produce a large amount of output."};
  };

  static constexpr Options::String help{
      "Sets a goal based on time-stepper truncation error."};
  using options = tmpl::list<AbsoluteTolerance, RelativeTolerance, MaxFactor,
                             MinFactor, SafetyFactor, Diagnostics>;

  ErrorControl(const double absolute_tolerance, const double relative_tolerance,
               const double max_factor, const double min_factor,
               const double safety_factor, const bool diagnostics = false)
      : absolute_tolerance_{absolute_tolerance},
        relative_tolerance_{relative_tolerance},
        max_factor_{max_factor},
        min_factor_{min_factor},
        safety_factor_{safety_factor},
        diagnostics_{diagnostics} {}

  using argument_tags = typename ErrorControl_detail::argument_tags<
      System, tmpl::list<::Tags::StepperErrors<VariablesTags>...>>::type;

  TimeStepRequest operator()(
      const typename ::Tags::StepperErrors<VariablesTags>::type&... errors,
      const double /*previous_step*/) const {
    const std::array goals{
        ErrorControl_detail::goal_from_variable<StepChooserUse>(
            errors, min_factor_, max_factor_, safety_factor_)...};
    std::optional<double> tightest_goal{};
    for (const auto& goal : goals) {
      if (goal.has_value() and (not tightest_goal.has_value() or
                                std::abs(*goal) < std::abs(*tightest_goal))) {
        tightest_goal = goal;
      }
    }
    return ::TimeStepRequest{.size_goal = tightest_goal};
  }

  template <typename LocalSystem = System,
            Requires<ErrorControl_detail::has_volume_dim<LocalSystem>::value> =
                nullptr>
  TimeStepRequest operator()(
      const typename ::Tags::StepperErrors<VariablesTags>::type&... errors,
      const Element<LocalSystem::volume_dim>& element,
      const TimeStepId& time_step_id, const double previous_step) const {
    const std::array goals{
        ErrorControl_detail::goal_from_variable<StepChooserUse>(
            errors, min_factor_, max_factor_, safety_factor_)...};
    std::optional<double> tightest_goal{};
    for (const auto& goal : goals) {
      if (goal.has_value() and (not tightest_goal.has_value() or
                                std::abs(*goal) < std::abs(*tightest_goal))) {
        tightest_goal = goal;
      }
    }

    if (diagnostics_) {
      size_t goal_index = 0;
      (write_diagnostics<VariablesTags>(errors, goals[goal_index++], element,
                                        time_step_id, previous_step),
       ...);
    }
    return ::TimeStepRequest{.size_goal = tightest_goal};
  }

  bool uses_local_data() const override { return true; }
  bool can_be_delayed() const override { return true; }
  bool must_set_step_size() const override { return true; }

  std::unordered_map<std::type_index, StepperErrorTolerances> tolerances()
      const override {
    return {{typeid(VariablesTags),
             {.estimates = StepperErrorTolerances::Estimates::StepperOrder,
              .absolute = absolute_tolerance_,
              .relative = relative_tolerance_,
              .diagnostics = diagnostics_}}...};
  }

  void pup(PUP::er& p) override {  // NOLINT
    StepChooser<StepChooserUse>::pup(p);
    p | absolute_tolerance_;
    p | relative_tolerance_;
    p | min_factor_;
    p | max_factor_;
    p | safety_factor_;
    p | diagnostics_;
  }

 private:
  template <typename VariablesTag, typename LocalSystem = System,
            Requires<ErrorControl_detail::has_volume_dim<LocalSystem>::value> =
                nullptr>
  void write_diagnostics(
      const std::array<std::optional<StepperErrorEstimate>, 2>& errors,
      const std::optional<double>& goal,
      const Element<LocalSystem::volume_dim>& element,
      const TimeStepId& time_step_id, const double previous_step) const {
    if (not errors[1].has_value()) {
      Parallel::fprintf("TimeStepperErrorDiagnostics.txt",
                        "ErrorControl element=%s evaluation_time=%.17g "
                        "variables=%s status=no_new_error_estimate "
                        "event_previous_step=%.17g atol=%.17g rtol=%.17g\n",
                        element.id(), time_step_id.step_time().value(),
                        db::tag_name<VariablesTag>(), previous_step,
                        absolute_tolerance_, relative_tolerance_);
      return;
    }

    const auto& current_error = *errors[1];
    const bool use_pi_controller =
        not std::is_same_v<StepChooserUse, ::StepChooserUse::LtsStep> and
        errors[0].has_value() and errors[0]->order == current_error.order;
    const double current_error_for_power =
        std::max(current_error.step_error(), 1e-14);
    double raw_factor = 0.0;
    if (use_pi_controller) {
      const double alpha_factor =
          0.7 / static_cast<double>(current_error.order + 1);
      const double beta_factor =
          0.4 / static_cast<double>(errors[0]->order + 1);
      raw_factor = safety_factor_ *
                   pow(1.0 / current_error_for_power, alpha_factor) *
                   pow(std::max(errors[0]->step_error(), 1e-14), beta_factor);
    } else {
      raw_factor = safety_factor_ *
                   pow(1.0 / current_error_for_power,
                       1.0 / static_cast<double>(current_error.order + 1));
    }
    const double applied_factor =
        std::clamp(raw_factor, min_factor_, max_factor_);
    const double previous_error =
        errors[0].has_value() ? errors[0]->step_error()
                              : std::numeric_limits<double>::quiet_NaN();

    const auto* const diagnostics = current_error.diagnostics.has_value()
                                        ? &current_error.diagnostics.value()
                                        : nullptr;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const size_t invalid_index = StepperErrorDiagnostics::invalid_index;
    const size_t number_of_values =
        diagnostics == nullptr ? 0 : diagnostics->number_of_values;
    const auto maximum_error_location =
        ErrorControl_detail::component_location<VariablesTag>(
            diagnostics == nullptr ? invalid_index
                                   : diagnostics->maximum_absolute_error_index,
            number_of_values);
    const auto maximum_start_value_location =
        ErrorControl_detail::component_location<VariablesTag>(
            diagnostics == nullptr
                ? invalid_index
                : diagnostics->maximum_absolute_step_start_value_index,
            number_of_values);
    const auto maximum_end_value_location =
        ErrorControl_detail::component_location<VariablesTag>(
            diagnostics == nullptr
                ? invalid_index
                : diagnostics->maximum_absolute_step_end_value_index,
            number_of_values);
    const auto maximum_pointwise_error_location =
        ErrorControl_detail::component_location<VariablesTag>(
            diagnostics == nullptr
                ? invalid_index
                : diagnostics->maximum_pointwise_normalized_error_index,
            number_of_values);
    constexpr double spec_style_epsilon = 1.0e-12;
    const double spec_style_local_relative_error =
        diagnostics == nullptr
            ? nan
            : diagnostics->maximum_absolute_error /
                  (diagnostics->maximum_absolute_step_end_value +
                   spec_style_epsilon);

    Parallel::fprintf(
        "TimeStepperErrorDiagnostics.txt",
        "ErrorControl element=%s evaluation_time=%.17g variables=%s "
        "error_step_start=%.17g error_step_size=%.17g error_order=%zu "
        "normalized_error=%.17g previous_normalized_error=%.17g "
        "controller=%s raw_factor=%.17g applied_factor=%.17g "
        "element_size_goal=%.17g event_previous_step=%.17g "
        "atol=%.17g rtol=%.17g safety=%.17g min_factor=%.17g "
        "max_factor=%.17g diagnostics_status=%s number_of_values=%zu "
        "number_of_grid_points=%zu "
        "max_abs_error=%.17g max_abs_error_flat_index=%zu "
        "max_abs_error_component=%s max_abs_error_component_index=%zu "
        "max_abs_error_grid_point=%zu "
        "max_abs_step_start_value=%.17g "
        "max_abs_step_start_value_flat_index=%zu "
        "max_abs_step_start_value_component=%s "
        "max_abs_step_start_value_component_index=%zu "
        "max_abs_step_start_value_grid_point=%zu "
        "max_abs_step_end_value=%.17g "
        "max_abs_step_end_value_flat_index=%zu "
        "max_abs_step_end_value_component=%s "
        "max_abs_step_end_value_component_index=%zu "
        "max_abs_step_end_value_grid_point=%zu "
        "spec_style_epsilon=%.17g spec_style_local_relative_error=%.17g "
        "max_pointwise_normalized_error=%.17g "
        "max_pointwise_normalized_error_flat_index=%zu "
        "max_pointwise_normalized_error_component=%s "
        "max_pointwise_normalized_error_component_index=%zu "
        "max_pointwise_normalized_error_grid_point=%zu "
        "abs_error_at_max_pointwise_normalized_error=%.17g "
        "abs_step_start_value_at_max_pointwise_normalized_error=%.17g "
        "abs_step_end_estimate_at_max_pointwise_normalized_error=%.17g "
        "normalization_at_max_pointwise_normalized_error=%.17g\n",
        element.id(), time_step_id.step_time().value(),
        db::tag_name<VariablesTag>(), current_error.step_time.value(),
        current_error.step_size.value(), current_error.order,
        current_error.step_error(), previous_error,
        use_pi_controller ? "PI" : "P", raw_factor, applied_factor,
        goal.value_or(std::numeric_limits<double>::quiet_NaN()), previous_step,
        absolute_tolerance_, relative_tolerance_, safety_factor_, min_factor_,
        max_factor_, diagnostics == nullptr ? "unavailable" : "available",
        number_of_values, maximum_error_location.number_of_grid_points,
        diagnostics == nullptr ? nan : diagnostics->maximum_absolute_error,
        diagnostics == nullptr ? invalid_index
                               : diagnostics->maximum_absolute_error_index,
        maximum_error_location.name,
        maximum_error_location.independent_component,
        maximum_error_location.grid_point,
        diagnostics == nullptr ? nan
                               : diagnostics->maximum_absolute_step_start_value,
        diagnostics == nullptr
            ? invalid_index
            : diagnostics->maximum_absolute_step_start_value_index,
        maximum_start_value_location.name,
        maximum_start_value_location.independent_component,
        maximum_start_value_location.grid_point,
        diagnostics == nullptr ? nan
                               : diagnostics->maximum_absolute_step_end_value,
        diagnostics == nullptr
            ? invalid_index
            : diagnostics->maximum_absolute_step_end_value_index,
        maximum_end_value_location.name,
        maximum_end_value_location.independent_component,
        maximum_end_value_location.grid_point, spec_style_epsilon,
        spec_style_local_relative_error,
        diagnostics == nullptr
            ? nan
            : diagnostics->maximum_pointwise_normalized_error,
        diagnostics == nullptr
            ? invalid_index
            : diagnostics->maximum_pointwise_normalized_error_index,
        maximum_pointwise_error_location.name,
        maximum_pointwise_error_location.independent_component,
        maximum_pointwise_error_location.grid_point,
        diagnostics == nullptr
            ? nan
            : diagnostics->absolute_error_at_maximum_pointwise_normalized_error,
        diagnostics == nullptr
            ? nan
            : diagnostics
                  ->absolute_step_start_value_at_maximum_pointwise_normalized_error,
        diagnostics == nullptr
            ? nan
            : diagnostics
                  ->absolute_step_end_estimate_at_maximum_pointwise_normalized_error,
        diagnostics == nullptr
            ? nan
            : diagnostics->normalization_at_maximum_pointwise_normalized_error);
  }

  double absolute_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  double relative_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  double max_factor_ = std::numeric_limits<double>::signaling_NaN();
  double min_factor_ = std::numeric_limits<double>::signaling_NaN();
  double safety_factor_ = std::numeric_limits<double>::signaling_NaN();
  bool diagnostics_ = false;
};
/// \cond
template <typename StepChooserUse, typename System, typename... VariablesTags>
PUP::able::PUP_ID
    ErrorControl<StepChooserUse, System,
                 tmpl::list<VariablesTags...>>::my_PUP_ID =  // NOLINT
    0;
/// \endcond
}  // namespace StepChoosers
