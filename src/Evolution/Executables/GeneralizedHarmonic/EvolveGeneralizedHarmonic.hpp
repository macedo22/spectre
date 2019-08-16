// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "AlgorithmSingleton.hpp"
#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Filtering.hpp"
#include "Evolution/DiscontinuousGalerkin/ObserveFields.hpp"
#include "Evolution/DiscontinuousGalerkin/ObserveNorms.hpp"
#include "Evolution/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"  // IWYU pragma: keep // for UpwindFlux
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/DataImporter/DataFileReader.hpp"
#include "IO/DataImporter/ElementActions.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "IO/Observer/Tags.hpp"  // IWYU pragma: keep
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolate.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"            // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"         // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"       // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                // IWYU pragma: keep
#include "Time/StepChoosers/Cfl.hpp"               // IWYU pragma: keep
#include "Time/StepChoosers/Constant.hpp"          // IWYU pragma: keep
#include "Time/StepChoosers/Increase.hpp"          // IWYU pragma: keep
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond

struct EvolutionMetavars {
  // Customization/"input options" to simulation
  static constexpr int dim = 3;
  using Inertial = Frame::Inertial;
  using system = GeneralizedHarmonic::System<dim>;
  using temporal_id = Tags::TimeId;
  static constexpr bool local_time_stepping = false;
  using analytic_solution_tag = OptionTags::AnalyticSolution<
      GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>>;
  using boundary_condition_tag = analytic_solution_tag;
  using normal_dot_numerical_flux = OptionTags::NumericalFlux<
      // dg::NumericalFluxes::LocalLaxFriedrichs<system>>;
      GeneralizedHarmonic::UpwindFlux<dim>>;

  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;
  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<domain_dim, domain_frame>,
                 GeneralizedHarmonic::Tags::Pi<domain_dim, domain_frame>,
                 GeneralizedHarmonic::Tags::Phi<domain_dim, domain_frame>,
                 gr::Tags::RicciTensor<domain_dim, domain_frame, DataVector>>;

  using observation_tags = tmpl::list<
      GeneralizedHarmonic::Tags::GaugeConstraint<domain_dim, domain_frame>,
      GeneralizedHarmonic::Tags::FConstraint<domain_dim, domain_frame>,
      GeneralizedHarmonic::Tags::TwoIndexConstraint<domain_dim, domain_frame>,
      GeneralizedHarmonic::Tags::ThreeIndexConstraint<domain_dim, domain_frame>,
      GeneralizedHarmonic::Tags::FourIndexConstraint<domain_dim, domain_frame>,
      GeneralizedHarmonic::Tags::ConstraintEnergy<domain_dim, domain_frame>,
      StrahlkorperGr::Tags::RicciScalar3D>;
  using observation_events = tmpl::list<
      dg::Events::Registrars::ObserveNorms<domain_dim, observation_tags>,
      dg::Events::Registrars::ObserveFields<domain_dim, observation_tags,
                                            tmpl::list<>>>;
  using events = tmpl::push_back<observation_events,
                                 intrp::Events::Registrars::Interpolate<
                                     domain_dim, interpolator_source_vars>>;
  using triggers = Triggers::time_triggers;

  // A tmpl::list of tags to be added to the ConstGlobalCache by the
  // metavariables
  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag,
                 OptionTags::TypedTimeStepper<tmpl::conditional_t<
                     local_time_stepping, LtsTimeStepper, TimeStepper>>,
                 OptionTags::EventsAndTriggers<events, triggers>>;
  using domain_creator_tag = OptionTags::DomainCreator<dim, Inertial>;

  using step_choosers = tmpl::list<StepChoosers::Registrars::Cfl<dim, Inertial>,
                                   StepChoosers::Registrars::Constant,
                                   StepChoosers::Registrars::Increase>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  // Horizon finding struct
  struct Horizon {
    using tags_to_observe = tmpl::list<
        StrahlkorperGr::Tags::SurfaceIntegral<StrahlkorperGr::Tags::Unity,
                                              domain_frame>,
        StrahlkorperGr::Tags::Area, StrahlkorperGr::Tags::IrreducibleMass,
        StrahlkorperGr::Tags::MaxRicciScalar,
        StrahlkorperGr::Tags::MinRicciScalar,
        StrahlkorperGr::Tags::MaxRicciScalar3D,
        StrahlkorperGr::Tags::MaxRicciScalar2ndTerm,
        StrahlkorperTags::MaxRadius<domain_frame>,
        StrahlkorperTags::MinRadius<domain_frame>>;
    using compute_items_on_source = tmpl::list<
        gr::Tags::SpatialMetricCompute<domain_dim, domain_frame, DataVector>,
        ah::Tags::InverseSpatialMetricCompute<domain_dim, domain_frame>,
        ah::Tags::ExtrinsicCurvatureCompute<domain_dim, domain_frame>,
        ah::Tags::SpatialChristoffelSecondKindCompute<domain_dim,
                                                      domain_frame>>;
    using vars_to_interpolate_to_target = tmpl::list<
        gr::Tags::SpatialMetric<domain_dim, domain_frame, DataVector>,
        gr::Tags::InverseSpatialMetric<domain_dim, domain_frame>,
        gr::Tags::ExtrinsicCurvature<domain_dim, domain_frame>,
        gr::Tags::SpatialChristoffelSecondKind<domain_dim, domain_frame>,
        gr::Tags::RicciTensor<domain_dim, domain_frame, DataVector>>;
    using compute_items_on_target = tmpl::list<
        StrahlkorperTags::MaxRadius<domain_frame>,
        StrahlkorperTags::MinRadius<domain_frame>,
        StrahlkorperTags::OneOverOneFormMagnitudeCompute<domain_frame>,
        StrahlkorperTags::UnitNormalOneFormCompute<domain_frame>,
        StrahlkorperTags::UnitNormalVectorCompute<domain_frame>,
        StrahlkorperGr::Tags::RicciScalarCompute<domain_frame>,
        StrahlkorperGr::Tags::MaxRicciScalarCompute,
        StrahlkorperGr::Tags::MinRicciScalarCompute,
        StrahlkorperGr::Tags::RicciScalar3DCompute<domain_frame>,
        StrahlkorperGr::Tags::MaxRicciScalar3DCompute,
        StrahlkorperGr::Tags::MinRicciScalar3DCompute,
        StrahlkorperGr::Tags::RicciScalar2ndTermCompute<domain_frame>,
        StrahlkorperGr::Tags::MaxRicciScalar2ndTermCompute,
        StrahlkorperGr::Tags::MinRicciScalar2ndTermCompute,
        StrahlkorperGr::Tags::AreaElement<domain_frame>,
        StrahlkorperGr::Tags::AreaCompute<domain_frame>,
        StrahlkorperGr::Tags::Unity,
        StrahlkorperGr::Tags::SurfaceIntegral<StrahlkorperGr::Tags::Unity,
                                              domain_frame>,
        StrahlkorperGr::Tags::IrreducibleMassCompute<domain_frame>,
        StrahlkorperTags::YlmSpherepackCompute<domain_frame>,
        StrahlkorperGr::Tags::SpinFunctionCompute<domain_frame>>;
    using compute_target_points =
        intrp::Actions::ApparentHorizon<Horizon, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<Horizon>;

    using post_horizon_find_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, Horizon,
                                                     Horizon>;
    // This `type` is so this tag can be used to read options.
    using type = typename compute_target_points::options_type;
    static constexpr OptionString help{
        "Options for interpolation onto Horizon.\n\n"};
  };
  using interpolation_target_tags = tmpl::list<Horizon>;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::push_back<Event<observation_events>::creatable_classes,
                      typename Horizon::post_horizon_find_callback>>;

  using compute_rhs = tmpl::flatten<tmpl::list<
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::InternalDirections<dim>>,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeTimeDerivative,
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::BoundaryDirectionsInterior<dim>>,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyFluxes>,
      // dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
      GeneralizedHarmonic::Actions::
          ImposeConstraintPreservingBoundaryConditions<EvolutionMetavars>,
      Actions::RecordTimeStepperData>>;
  using update_variables = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<local_time_stepping,
                          dg::Actions::ApplyBoundaryFluxesLocalTimeStepping,
                          tmpl::list<>>,
      Actions::UpdateU,
      dg::Actions::ExponentialFilter<
          0, typename system::variables_tag::type::tags_list>>>;

  using import_fields = tmpl::list<
      gr::Tags::SpacetimeMetric<dim, Inertial, DataVector>,
      GeneralizedHarmonic::Tags::Pi<dim, Inertial>,
      GeneralizedHarmonic::Tags::Phi<dim, Inertial>,
      GeneralizedHarmonic::Tags::InitialGaugeH<dim, Inertial>,
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<dim, Inertial>>;

  enum class Phase {
    Initialization,
    InitializeTimeStepperHistory,
    RegisterWithObserver,
    ImportData,
    Evolve,
    Exit
  };

  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      importer::DataFileReader<EvolutionMetavars>,
      intrp::Interpolator<EvolutionMetavars>,
      intrp::InterpolationTarget<EvolutionMetavars, Horizon>,
      DgElementArray<
          EvolutionMetavars,
          tmpl::list<
              Parallel::PhaseActions<
                  Phase, Phase::Initialization,
                  tmpl::list<GeneralizedHarmonic::Actions::Initialize<dim>>>,
              Parallel::PhaseActions<
                  Phase, Phase::InitializeTimeStepperHistory,
                  tmpl::flatten<tmpl::list<SelfStart::self_start_procedure<
                      compute_rhs, update_variables>>>>,
              Parallel::PhaseActions<
                  Phase, Phase::RegisterWithObserver,
                  tmpl::list<Actions::AdvanceTime,
                             intrp::Actions::RegisterElementWithInterpolator,
                             observers::Actions::RegisterWithObservers<
                                 observers::RegisterObservers<
                                     element_observation_type>>,
                             importer::Actions::RegisterWithImporter,
                             Parallel::Actions::TerminatePhase>>,
              Parallel::PhaseActions<
                  Phase, Phase::Evolve,
                  tmpl::flatten<tmpl::list<
                      Actions::RunEventsAndTriggers,
                      tmpl::conditional_t<
                          local_time_stepping,
                          Actions::ChangeStepSize<step_choosers>, tmpl::list<>>,
                      compute_rhs, update_variables, Actions::AdvanceTime>>>>,
          GeneralizedHarmonic::Actions::Initialize<dim>::AddOptionsToDataBox,
          import_fields>>;

  static constexpr OptionString help{
      "Evolve a generalized harmonic analytic solution.\n\n"
      "The analytic solution is: KerrSchild\n"
      "The numerical flux is:    UpwindFlux\n"};

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::ImportData;
      case Phase::ImportData:
        return Phase::Evolve;
      case Phase::Evolve:
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
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<MathFunction<1>>,
    &Parallel::register_derived_classes_with_charm<
         Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
         StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
         Trigger<metavariables::triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
