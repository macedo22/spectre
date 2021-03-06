// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeBackgroundFields.hpp"
#include "Elliptic/Actions/InitializeFields.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/NewtonRaphson.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace SolveXcts::OptionTags {
struct NonlinearSolverGroup {
  static std::string name() noexcept { return "NonlinearSolver"; }
  static constexpr Options::String help = "The iterative nonlinear solver";
};
struct NewtonRaphsonGroup {
  static std::string name() noexcept { return "NewtonRaphson"; }
  static constexpr Options::String help =
      "Options for the Newton-Raphson nonlinear solver";
  using group = NonlinearSolverGroup;
};
struct LinearSolverGroup {
  static std::string name() noexcept { return "LinearSolver"; }
  static constexpr Options::String help =
      "The iterative Krylov-subspace linear solver";
};
struct GmresGroup {
  static std::string name() noexcept { return "Gmres"; }
  static constexpr Options::String help = "Options for the GMRES linear solver";
  using group = LinearSolverGroup;
};
}  // namespace SolveXcts::OptionTags

/// \cond
struct Metavariables {
  static constexpr size_t volume_dim = 3;
  static constexpr int conformal_matter_scale = 0;
  using system =
      Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift,
                             Xcts::Geometry::Curved, conformal_matter_scale>;

  // List the possible backgrounds, i.e. the variable-independent part of the
  // equations that define the problem to solve (along with the boundary
  // conditions)
  using analytic_solution_registrars =
      tmpl::list<Xcts::Solutions::Registrars::Schwarzschild,
                 Xcts::Solutions::Registrars::Kerr>;
  using analytic_data_registrars = tmpl::list<
      Xcts::AnalyticData::Registrars::Binary<analytic_solution_registrars>>;
  using background_tag = elliptic::Tags::Background<
      ::AnalyticData<volume_dim, tmpl::append<analytic_solution_registrars,
                                              analytic_data_registrars>>>;

  // List the possible initial guesses
  using initial_guess_registrars =
      tmpl::append<tmpl::list<Xcts::Solutions::Registrars::Flatness>,
                   analytic_solution_registrars, analytic_data_registrars>;
  using initial_guess_tag = elliptic::Tags::InitialGuess<
      ::AnalyticData<volume_dim, initial_guess_registrars>>;

  static constexpr Options::String help{
      "Find the solution to an XCTS problem."};

  // These are the fields we solve for
  using fields_tag = ::Tags::Variables<typename system::primal_fields>;
  // These are the fluxes corresponding to the fields, i.e. essentially their
  // first derivatives. These are background fields for the linearized sources.
  using fluxes_tag = ::Tags::Variables<typename system::primal_fluxes>;
  // These are the fixed sources, i.e. the RHS of the equations
  using fixed_sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, fields_tag>;

  using nonlinear_solver = NonlinearSolver::newton_raphson::NewtonRaphson<
      Metavariables, fields_tag, SolveXcts::OptionTags::NewtonRaphsonGroup,
      fixed_sources_tag>;
  using nonlinear_solver_iteration_id =
      Convergence::Tags::IterationId<typename nonlinear_solver::options_group>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not guaranteed to be symmetric.
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, typename nonlinear_solver::linear_solver_fields_tag,
      SolveXcts::OptionTags::GmresGroup, false,
      typename nonlinear_solver::linear_solver_source_tag>;
  using linear_solver_iteration_id =
      Convergence::Tags::IterationId<typename linear_solver::options_group>;
  // For the GMRES linear solver we need to apply the DG operator to its
  // internal "operand" in every iteration of the algorithm.
  using correction_vars_tag = typename linear_solver::operand_tag;
  using operator_applied_to_correction_vars_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                         correction_vars_tag>;
  // The correction fluxes can be stored in an arbitrary tag. We don't need to
  // access them anywhere, they're just a memory buffer for the linearized
  // operator.
  using correction_fluxes_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fluxes_tag>;

  // Collect events and triggers
  // (public for use by the Charm++ registration code)
  using analytic_solution_fields = tmpl::append<typename system::primal_fields,
                                                typename system::primal_fluxes>;
  using observe_fields = tmpl::append<analytic_solution_fields,
                                      typename system::background_fields>;
  using events =
      tmpl::list<dg::Events::Registrars::ObserveFields<
                     volume_dim, nonlinear_solver_iteration_id, observe_fields,
                     analytic_solution_fields>,
                 dg::Events::Registrars::ObserveErrorNorms<
                     nonlinear_solver_iteration_id, analytic_solution_fields>>;

  // Collect all items to store in the cache.
  using const_global_cache_tags = tmpl::list<background_tag, initial_guess_tag,
                                             Tags::EventsAndTriggers<events>>;

  // Collect all reduction tags for observers
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::flatten<tmpl::list<typename Event<events>::creatable_classes,
                               nonlinear_solver, linear_solver>>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        Trigger, tmpl::push_back<Triggers::logical_triggers,
                                 elliptic::Triggers::EveryNIterations<
                                     nonlinear_solver_iteration_id>>>>;
  };

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

  using initialization_actions = tmpl::list<
      Actions::SetupDataBox,
      elliptic::dg::Actions::InitializeDomain<volume_dim>,
      typename nonlinear_solver::initialize_element,
      typename linear_solver::initialize_element,
      elliptic::Actions::InitializeFields<system, initial_guess_tag>,
      elliptic::Actions::InitializeFixedSources<system, background_tag>,
      elliptic::Actions::InitializeBackgroundFields<system, background_tag>,
      elliptic::Actions::InitializeOptionalAnalyticSolution<
          background_tag, analytic_solution_fields,
          Xcts::Solutions::AnalyticSolution<tmpl::append<
              analytic_solution_registrars, analytic_data_registrars>>>,
      elliptic::dg::Actions::initialize_operator<system>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  template <bool Linearized>
  using build_operator_actions = elliptic::dg::Actions::apply_operator<
      system, Linearized,
      tmpl::conditional_t<Linearized, linear_solver_iteration_id,
                          nonlinear_solver_iteration_id>,
      tmpl::conditional_t<Linearized, correction_vars_tag, fields_tag>,
      tmpl::conditional_t<Linearized, correction_fluxes_tag, fluxes_tag>,
      tmpl::conditional_t<Linearized, operator_applied_to_correction_vars_tag,
                          operator_applied_to_fields_tag>>;

  using register_actions =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 typename nonlinear_solver::register_element,
                 Parallel::Actions::TerminatePhase>;

  using solve_actions = tmpl::list<
      typename nonlinear_solver::template solve<
          build_operator_actions<false>,
          typename linear_solver::template solve<build_operator_actions<true>>,
          Actions::RunEventsAndTriggers>,
      Parallel::Actions::TerminatePhase>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<Parallel::PhaseActions<Phase, Phase::Initialization,
                                        initialization_actions>,
                 Parallel::PhaseActions<Phase, Phase::RegisterWithObserver,
                                        register_actions>,
                 Parallel::PhaseActions<Phase, Phase::Solve, solve_actions>>>;

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename nonlinear_solver::component_list,
                 typename linear_solver::component_list,
                 observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>>;

  // Specify the transitions between phases.
  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<
          tuples::TaggedTuple<Tags...>*> /*phase_change_decision_data*/,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Solve;
      case Phase::Solve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        metavariables::background_tag::type::element_type>,
    &Parallel::register_derived_classes_with_charm<
        metavariables::initial_guess_tag::type::element_type>,
    &Parallel::register_derived_classes_with_charm<
        Xcts::Solutions::AnalyticSolution<
            typename metavariables::analytic_solution_registrars>>,
    &Parallel::register_derived_classes_with_charm<
        metavariables::system::boundary_conditions_base>,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
