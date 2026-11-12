// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Time/LtsMode.hpp"
#include "Time/OptionTags/LocalTimeStepping.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace evolution::dg::stage_rhs_diagnostics {

/// A Cartesian product of component names and flattened grid-point indices
/// in one element. Names use Tensor::component_suffix, as in ErrorControl's
/// existing diagnostics (e.g. Pi_zt, not Pi_tz).
struct Target {
  struct Element {
    using type = std::string;
    static constexpr Options::String help{
        "Exact element ID, including brackets."};
  };
  struct Components {
    using type = std::vector<std::string>;
    static constexpr Options::String help{"Evolved component names to sample."};
  };
  struct GridPoints {
    using type = std::vector<size_t>;
    static constexpr Options::String help{
        "Zero-based flattened grid-point indices."};
  };
  using options = tmpl::list<Element, Components, GridPoints>;
  static constexpr Options::String help{
      "Points and components in one element."};

  Target() = default;
  Target(const std::string& element_in, std::vector<std::string> components_in,
         std::vector<size_t> points_in, const Options::Context& context = {})
      : element(element_in),
        components(std::move(components_in)),
        points(std::move(points_in)) {
    if (components.empty() or points.empty()) {
      PARSE_ERROR(context, "Components and GridPoints must be nonempty.");
    }
    for (size_t i = 0; i < components.size(); ++i) {
      if (std::find(components.begin(), components.begin() + i,
                    components[i]) != components.begin() + i) {
        PARSE_ERROR(context,
                    "Repeated component in StageRhsDiagnostics target.");
      }
    }
    for (size_t i = 0; i < points.size(); ++i) {
      if (std::find(points.begin(), points.begin() + i, points[i]) !=
          points.begin() + i) {
        PARSE_ERROR(context,
                    "Repeated grid point in StageRhsDiagnostics target.");
      }
    }
  }
  void pup(PUP::er& p) {
    p | element;
    p | components;
    p | points;
  }
  ElementId<3> element{};
  std::vector<std::string> components{};
  std::vector<size_t> points{};
};

struct Settings {
  struct StartTime {
    using type = double;
    static constexpr Options::String help{
        "Include steps starting at or after this time."};
  };
  struct EndTime {
    using type = double;
    static constexpr Options::String help{
        "Exclude steps starting at or after this time."};
  };
  struct Targets {
    using type = std::vector<Target>;
    static constexpr Options::String help{
        "Elements, components, and points to sample."};
  };
  using options = tmpl::list<StartTime, EndTime, Targets>;
  static constexpr Options::String help{
      "Targeted GH/GTS DormandPrince5 stage diagnostics. Writes "
      "TimeStepperStageDiagnostics.txt; independent of "
      "ErrorControl.Diagnostics."};

  Settings() = default;
  Settings(const double start, const double end, std::vector<Target> targets_in,
           const Options::Context& context = {})
      : start_time(start), end_time(end), targets(std::move(targets_in)) {
    if (not std::isfinite(start) or not std::isfinite(end) or end <= start or
        targets.empty()) {
      PARSE_ERROR(context,
                  "Require finite StartTime < EndTime and nonempty Targets.");
    }
    for (size_t i = 0; i < targets.size(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        if (targets[i].element == targets[j].element) {
          PARSE_ERROR(context,
                      "Combine selections for an element in one Target.");
        }
      }
    }
  }
  void pup(PUP::er& p) {
    p | start_time;
    p | end_time;
    p | targets;
  }
  double start_time = 0.0;
  double end_time = 0.0;
  std::vector<Target> targets{};
};

namespace OptionTags {
struct StageRhsDiagnostics {
  using type = Options::Auto<Settings, Options::AutoLabel::None>;
  static constexpr Options::String help{
      "Targeted stage/RHS diagnostics for EvolveGhBinaryBlackHole. None "
      "disables "
      "all sampling and output. Requires GTS DormandPrince5 with error "
      "estimates."};
};
}  // namespace OptionTags

struct Stage {
  double time = 0.0;
  double value = 0.0;
  double volume = 0.0;
  double mesh = 0.0;
  double total = 0.0;
  std::array<double, 3> coordinates{};
  std::array<double, 3> velocity{};
  bool captured = false;
  bool recorded = false;
  void pup(PUP::er& p) {
    p | time;
    p | value;
    p | volume;
    p | mesh;
    p | total;
    p | coordinates;
    p | velocity;
    p | captured;
    p | recorded;
  }
};

struct Sample {
  std::string component{};
  size_t component_index = 0;
  size_t point = 0;
  std::array<Stage, 7> stages{};
  void pup(PUP::er& p) {
    p | component;
    p | component_index;
    p | point;
    p | stages;
  }
};

struct State {
  std::optional<Settings> settings{};
  std::optional<TimeStepId> step_start{};
  std::array<size_t, 3> extents{};
  std::vector<Sample> samples{};
  void pup(PUP::er& p) {
    p | settings;
    p | step_start;
    p | extents;
    p | samples;
  }
};

namespace Tags {
struct Data : db::SimpleTag {
  using type = State;
  using option_tags = tmpl::list<OptionTags::StageRhsDiagnostics,
                                 ::OptionTags::LocalTimeStepping>;
  static constexpr bool pass_metavariables = false;
  static State create_from_options(const std::optional<Settings>& settings,
                                   const LtsMode lts_mode) {
    if (settings.has_value() and lts_mode != LtsMode::Off) {
      ERROR_NO_TRACE("StageRhsDiagnostics requires LocalTimeStepping: Off.");
    }
    State state{};
    state.settings = settings;
    return state;
  }
};
}  // namespace Tags

struct Initialize {
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<Tags::Data>;
  using simple_tags = tmpl::list<>;
  using compute_tags = tmpl::list<>;
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;
  static void apply() {}
};

/// Preserve the options but discard samples on AMR. Raw point indices always
/// refer to the current mesh; users should use fixed meshes for comparisons.
struct Project : tt::ConformsTo<amr::protocols::Projector> {
  using return_tags = tmpl::list<Tags::Data>;
  using argument_tags = tmpl::list<>;

  static void apply(const gsl::not_null<State*> state,
                    const std::pair<Mesh<3>, Element<3>>& /*old_mesh*/) {
    auto settings = std::move(state->settings);
    *state = State{};
    state->settings = std::move(settings);
  }
  template <typename... ParentTags>
  static void apply(const gsl::not_null<State*> state,
                    const tuples::TaggedTuple<ParentTags...>& parent) {
    *state = State{};
    state->settings = tuples::get<Tags::Data>(parent).settings;
  }
  template <typename... ChildrenTags>
  static void apply(
      const gsl::not_null<State*> state,
      const std::unordered_map<
          ElementId<3>, tuples::TaggedTuple<ChildrenTags...>>& children) {
    *state = State{};
    state->settings =
        tuples::get<Tags::Data>(children.begin()->second).settings;
  }
};

inline bool in_window(const State& state, const TimeStepId& id) {
  return state.settings.has_value() and
         id.step_time().value() >= state.settings->start_time and
         id.step_time().value() < state.settings->end_time;
}

/// Called immediately after the volume terms, before any external or internal
/// boundary corrections. Only executables adding Tags::Data instantiate this
/// path. No new derivatives are computed: use the volume routine's derivatives.
template <typename System, typename DbTags, typename Derivatives>
void capture_volume(db::DataBox<DbTags>& box, const Derivatives& derivatives) {
  if constexpr (tmpl::list_contains_v<DbTags, Tags::Data>) {
    static_assert(System::volume_dim == 3);
    static_assert(tmpl::size<typename System::flux_variables>::value == 0,
                  "This diagnostic splits nonconservative volume derivatives.");
    using VarsTag = typename System::variables_tag;
    const auto& state = db::get<Tags::Data>(box);
    if (not state.settings.has_value()) {
      return;
    }
    const auto& id = db::get<::Tags::TimeStepId>(box);
    if (not in_window(state, id)) {
      return;
    }
    const auto& element = db::get<domain::Tags::Element<3>>(box);
    const auto target = std::find_if(
        state.settings->targets.begin(), state.settings->targets.end(),
        [&element](const Target& t) { return t.element == element.id(); });
    if (target == state.settings->targets.end()) {
      return;
    }
    if (dynamic_cast<const TimeSteppers::DormandPrince5*>(
            &db::get<::Tags::TimeStepper<TimeStepper>>(box)) == nullptr or
        db::get<::Tags::StepperErrorTolerances<VarsTag>>(box).estimates ==
            StepperErrorTolerances::Estimates::None) {
      ERROR_NO_TRACE(
          "StageRhsDiagnostics requires DormandPrince5 and "
          "ErrorControl requesting an embedded error estimate.");
    }
    const auto& mesh = db::get<domain::Tags::Mesh<3>>(box);
    const auto& vars = db::get<VarsTag>(box);
    const auto& dt_vars = db::get<db::add_tag_prefix<::Tags::dt, VarsTag>>(box);
    const auto& velocity = db::get<domain::Tags::MeshVelocity<3>>(box);
    const auto& coords =
        db::get<domain::Tags::Coordinates<3, Frame::Inertial>>(box);
    // Copy the target before mutating the State that owns it.
    const Target selection = *target;
    db::mutate<Tags::Data>(
        [&](const gsl::not_null<State*> data) {
          const size_t stage = id.substep();
          if (stage >= 7) {
            ERROR("Unexpected DormandPrince5 stage " << stage);
          }
          if (stage == 0) {
            data->step_start = id;
            data->samples.clear();
            for (size_t d = 0; d < 3; ++d) {
              data->extents[d] = mesh.extents(d);
            }
            size_t offset = 0;
            tmpl::for_each<typename VarsTag::tags_list>([&](auto tag_v) {
              using Tag = tmpl::type_from<decltype(tag_v)>;
              using Tensor = typename Tag::type;
              for (size_t c = 0; c < Tensor::size(); ++c) {
                const std::string name =
                    db::tag_name<Tag>() + Tensor::component_suffix(c);
                if (std::find(selection.components.begin(),
                              selection.components.end(),
                              name) != selection.components.end()) {
                  for (const size_t point : selection.points) {
                    if (point >= mesh.number_of_grid_points()) {
                      ERROR_NO_TRACE("StageRhsDiagnostics point "
                                     << point << " is outside " << element.id()
                                     << " with extents " << mesh.extents());
                    }
                    data->samples.push_back(
                        Sample{name, offset + c, point, {}});
                  }
                }
              }
              offset += Tensor::size();
            });
            for (const auto& name : selection.components) {
              if (std::none_of(data->samples.begin(), data->samples.end(),
                               [&name](const Sample& s) {
                                 return s.component == name;
                               })) {
                ERROR_NO_TRACE("Unknown StageRhsDiagnostics component '"
                               << name
                               << "'. Use the component spelling from "
                                  "TimeStepperErrorDiagnostics.txt.");
              }
            }
          }
          if (not data->step_start.has_value() or
              data->step_start->step_time() != id.step_time()) {
            ERROR_NO_TRACE(
                "StageRhsDiagnostics is missing stage zero. "
                "Start the diagnostic interval on a complete step.");
          }
          for (size_t d = 0; d < 3; ++d) {
            if (data->extents[d] != mesh.extents(d)) {
              ERROR_NO_TRACE(
                  "StageRhsDiagnostics mesh changed within an RK step.");
            }
          }
          for (auto& sample : data->samples) {
            auto& s = sample.stages[stage];
            s = Stage{};
            s.time = id.substep_time();
            const size_t flat =
                sample.component_index * mesh.number_of_grid_points() +
                sample.point;
            s.value = vars.data()[flat];
            s.volume = dt_vars.data()[flat];
            for (size_t d = 0; d < 3; ++d) {
              s.coordinates[d] = coords.get(d)[sample.point];
              s.velocity[d] =
                  velocity.has_value() ? velocity->get(d)[sample.point] : 0.0;
            }
            size_t offset = 0;
            tmpl::for_each<typename VarsTag::tags_list>([&](auto tag_v) {
              using Tag = tmpl::type_from<decltype(tag_v)>;
              using Tensor = typename Tag::type;
              if (sample.component_index >= offset and
                  sample.component_index < offset + Tensor::size()) {
                const auto tensor_index = get<Tag>(vars).get_tensor_index(
                    sample.component_index - offset);
                const auto& deriv =
                    get<::Tags::deriv<Tag, tmpl::size_t<3>, Frame::Inertial>>(
                        derivatives);
                for (size_t d = 0; d < 3; ++d) {
                  std::array<size_t, Tensor::rank() + 1> deriv_index{};
                  deriv_index[0] = d;
                  for (size_t j = 0; j < Tensor::rank(); ++j) {
                    deriv_index[j + 1] = tensor_index[j];
                  }
                  s.mesh +=
                      s.velocity[d] * deriv.get(deriv_index)[sample.point];
                }
              }
              offset += Tensor::size();
            });
            s.captured = true;
          }
        },
        make_not_null(&box));
  } else {
    (void)box;
    (void)derivatives;
  }
}

/// Run after UpdateU, before CleanHistory. The recorded derivative is precisely
/// the derivative used by the stepper. Subtract the pre-boundary volume sample
/// to get the total boundary contribution, including external boundaries.
template <typename System>
struct Observe {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* /*meta*/) {
    using VarsTag = typename System::variables_tag;
    const auto& state = db::get<Tags::Data>(box);
    if (not state.settings.has_value()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const auto& id = db::get<::Tags::TimeStepId>(box);
    if (not in_window(state, id) or state.samples.empty() or
        not state.step_start.has_value() or
        state.step_start->step_time() != id.step_time()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const auto& history =
        db::get<::Tags::HistoryEvolvedVariables<VarsTag>>(box);
    const auto& element = db::get<domain::Tags::Element<3>>(box);
    const double dt = db::get<::Tags::TimeStep>(box).value();
    const auto& tableau = dynamic_cast<const TimeSteppers::DormandPrince5&>(
                              db::get<::Tags::TimeStepper<TimeStepper>>(box))
                              .butcher_tableau();
    const auto& tolerances =
        db::get<::Tags::StepperErrorTolerances<VarsTag>>(box);
    const auto& errors = db::get<::Tags::StepperErrors<VarsTag>>(box);
    const auto& updated_vars = db::get<VarsTag>(box);
    db::mutate<Tags::Data>(
        [&](const gsl::not_null<State*> data) {
          const size_t stage = id.substep();
          if (stage >= 7) {
            ERROR("Unexpected DormandPrince5 stage " << stage);
          }
          const auto& record =
              stage == 0 ? history.back() : history.substeps()[stage - 1];
          if (record.time_step_id != id) {
            ERROR(
                "StageRhsDiagnostics history does not match the sampled "
                "stage.");
          }
          const size_t npoints = record.derivative.number_of_grid_points();
          const auto weight = [&tableau](const size_t i) {
            return (i < tableau.result_coefficients.size()
                        ? tableau.result_coefficients[i]
                        : 0.0) -
                   tableau.error_coefficients[i];
          };
          for (auto& sample : data->samples) {
            auto& s = sample.stages[stage];
            if (not s.captured) {
              ERROR("StageRhsDiagnostics missing volume sample.");
            }
            const size_t flat = sample.component_index * npoints + sample.point;
            s.total = record.derivative.data()[flat];
            s.recorded = true;
            const double inertial_volume = s.volume - s.mesh;
            const double boundary = s.total - s.volume;
            Parallel::fprintf(
                "TimeStepperStageDiagnostics.txt",
                "StageRhs element=%s step_start=%.17g dt=%.17g stage=%zu "
                "stage_time=%.17g component=%s component_index=%zu point=%zu "
                "nr=%zu ntheta=%zu nz=%zu value=%.17g "
                "x=%.17g y=%.17g z=%.17g vx=%.17g vy=%.17g vz=%.17g "
                "k_total=%.17g k_volume=%.17g k_inertial_volume=%.17g "
                "k_mesh=%.17g k_boundary=%.17g error_weight=%.17g "
                "error_contribution=%.17g\n",
                element.id(), id.step_time().value(), dt, stage, s.time,
                sample.component, sample.component_index, sample.point,
                data->extents[0], data->extents[1], data->extents[2], s.value,
                s.coordinates[0], s.coordinates[1], s.coordinates[2],
                s.velocity[0], s.velocity[1], s.velocity[2], s.total, s.volume,
                inertial_volume, s.mesh, boundary, weight(stage),
                weight(stage) * dt * s.total);
            if (stage != 6) {
              continue;
            }
            if (not errors[1].has_value() or
                errors[1]->step_time != id.step_time()) {
              ERROR("StageRhsDiagnostics missing completed embedded estimate.");
            }
            double delta_history = 0.0;
            double delta_sampled = 0.0;
            double delta_volume = 0.0;
            double delta_mesh = 0.0;
            double delta_boundary = 0.0;
            for (size_t i = 0; i < 7; ++i) {
              const auto& si = sample.stages[i];
              if (not si.recorded) {
                ERROR("StageRhsDiagnostics missing stage " << i);
              }
              if (weight(i) == 0.0) {
                continue;
              }
              const double coefficient = weight(i) * dt;
              const auto& ki = i == 0 ? history.back().derivative
                                      : history.substeps()[i - 1].derivative;
              delta_history += coefficient * ki.data()[flat];
              delta_sampled += coefficient * si.total;
              delta_volume += coefficient * (si.volume - si.mesh);
              delta_mesh += coefficient * si.mesh;
              delta_boundary += coefficient * (si.total - si.volume);
            }
            const double start_value = history.back().value->data()[flat];
            const double normalization =
                tolerances.absolute +
                tolerances.relative *
                    std::max(std::abs(start_value),
                             std::abs(start_value + delta_history));
            Parallel::fprintf(
                "TimeStepperStageDiagnostics.txt",
                "StageError element=%s step_start=%.17g dt=%.17g "
                "component=%s component_index=%zu point=%zu "
                "delta_history=%.17g delta_sampled=%.17g "
                "delta_inertial_volume=%.17g delta_mesh=%.17g "
                "delta_boundary=%.17g history_residual=%.17g "
                "split_residual=%.17g start_value=%.17g end_value=%.17g "
                "normalization=%.17g normalized_error=%.17g "
                "element_normalized_error=%.17g\n",
                element.id(), id.step_time().value(), dt, sample.component,
                sample.component_index, sample.point, delta_history,
                delta_sampled, delta_volume, delta_mesh, delta_boundary,
                delta_sampled - delta_history,
                delta_volume + delta_mesh + delta_boundary - delta_history,
                start_value, updated_vars.data()[flat], normalization,
                std::abs(delta_history) / normalization,
                errors[1]->step_error());
          }
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::stage_rhs_diagnostics
