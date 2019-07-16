// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "AlgorithmArray.hpp"
#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Initialization/Domain.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Time.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Actions {
//double overall_min_grid_spacing = std::numeric_limits<double>::max();
template <size_t Dim>
struct InitializeElement {
  struct InitialExtents : db::SimpleTag {
    static std::string name() noexcept { return "InitialExtents"; }
    using type = std::vector<std::array<size_t, Dim>>;
  };
  struct Domain : db::SimpleTag {
    static std::string name() noexcept { return "Domain"; }
    using type = ::Domain<Dim, Frame::Inertial>;
  };

  using AddOptionsToDataBox =
      Parallel::ForwardAllOptionsToDataBox<tmpl::list<InitialExtents, Domain>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<tmpl::list_contains_v<DbTagsList, Domain>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto initial_extents = db::get<InitialExtents>(box);
    ::Domain<Dim, Frame::Inertial> domain{};
    db::mutate<Domain>(
        make_not_null(&box), [&domain](const auto domain_ptr) noexcept {
          domain = std::move(*domain_ptr);
        });
    return std::make_tuple(
    //const auto& initial_box =
        Elliptic::Initialization::Domain<Dim>::initialize(
            db::create_from<typename AddOptionsToDataBox::simple_tags>(
                std::move(box)),
            array_index, initial_extents, domain),
        true);

    /*const auto& mesh = get<Tags::Mesh<Dim>>(initial_box);
    const auto& inertial_coordinates =
        db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(initial_box);

    const auto& box_with_min_spacing =
        db::create_from<
            db::RemoveTags<>,
            db::AddSimpleTags<>,
            db::AddComputeTags<Tags::MinimumGridSpacing<Dim, Frame::Inertial>>>(
        initial_box);
    return std::make_tuple(std::move(box_with_min_spacing), true);*/
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<not tmpl::list_contains_v<DbTagsList, Domain>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box), true};
  }
};

template <size_t Dim>
struct ExportCoordinates {
  // Compile-time interface for observers
  struct ObservationType {};
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationId>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    return {observers::TypeOfObservation::ReductionAndVolume,
            observers::ObservationId(0., ObservationType{})};
  }

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, Tags::Mesh<Dim>>> = nullptr>
  //static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
  static auto apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& mesh = get<Tags::Mesh<Dim>>(box);
    const auto& inertial_coordinates =
        db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';

    // Collect volume data
    // Remove tensor types, only storing individual components
    std::vector<TensorComponent> components;
    components.reserve(Dim);
    for (size_t d = 0; d < Dim; d++) {
      components.emplace_back(element_name + "InertialCoordinates_" +
                                  inertial_coordinates.component_name(
                                      inertial_coordinates.get_tensor_index(d)),
                              inertial_coordinates.get(d));
    }
    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer, observers::ObservationId(0., ObservationType{}),
        std::string{"/element_data"},
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementIndex<Dim>>(array_index)),
        std::move(components), mesh.extents());

    /*const double element_min_grid_spacing =
        get<Tags::MinimumGridSpacing<Dim, Frame::Inertial>>(
            box);*///box_with_min_spacing);


    /*if (element_min_grid_spacing < overall_min_grid_spacing) {
        overall_min_grid_spacing = element_min_grid_spacing;
    }*/

    /*printf("-----------Current minimum grid spacing status-----------\n");
    printf("Element's inertial minimum grid spacing calculated: %f\n",
      element_min_grid_spacing);
    printf("Overall inertial minimum grid spacing: %f\n\n",
      overall_min_grid_spacing);*/

    //return {std::move(box), true};
    return std::forward_as_tuple(std::move(box), true);
    //return std::forward_as_tuple(std::move(box_with_min_spacing), true);
  }
};

struct PrintMinimumGridSpacing {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const double& overall_min_grid_spacing) noexcept {
    /*SPECTRE_PARALLEL_REQUIRE(number_of_1d_array_elements *
                                 (number_of_1d_array_elements - 1) / 2 ==
                             value);*/
    printf("Overall inertial minimum grid spacing: %f\n\n",
              overall_min_grid_spacing);
  }
};

}  // namespace Actions

//template <typename Metavariables>
template <size_t Dim, typename Metavariables>
struct SingletonParallelComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;
  using metavariables = Metavariables;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& /*global_cache*/) {}

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) {}
};

template <size_t Dim, typename Metavariables>
struct ElementArray;

template <size_t Dim>
struct ArrayReduce {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTags, Tags::Mesh<Dim>>> = nullptr>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) noexcept {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   ElementArray<Dim, Metavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    const auto& my_proxy =
        Parallel::get_parallel_component<ElementArray<Dim, Metavariables>>(
            cache)[array_index];
    const auto& singleton_proxy = Parallel::get_parallel_component<
        SingletonParallelComponent<Dim, Metavariables>>(cache);

    Parallel::ReductionData<Parallel::ReductionDatum<double, funcl::Min<>>>
        elemental_min_grid_spacing{
            get<Tags::MinimumGridSpacing<Dim, Frame::Inertial>>(box)};
    Parallel::contribute_to_reduction<Actions::PrintMinimumGridSpacing>(
        elemental_min_grid_spacing, my_proxy, singleton_proxy);
  }
};

// A parallel component struct
template <size_t Dim, typename Metavariables>
struct ElementArray {
  // Hold elements distributed to some core
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using elemental_min_grid_spacing = double;
  // Type that indexes the Parallel Component Array
  using array_index = ElementIndex<Dim>;
  // OptionTags required by parallel component, usually obtained from
  // phase_dependent_action_list
  using const_global_cache_tag_list = tmpl::list<>;
  // list of the option structures. The options are read in from the input file
  // and passed to the initialize function of the parallel component. These
  // options are only used during initialization.
  using options = tmpl::list<OptionTags::DomainCreator<Dim, Frame::Inertial>>;
  // list of Parallel::PhaseActions<PhaseType, Phase, tmpl::list<Actions...>>
  // where each PhaseAction is a PDAL executed on the parallel component during
  // the specified phase. Actions executed in the order provided.
  using phase_dependent_action_list = tmpl::list<
      // Initialization phase
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<Actions::InitializeElement<Dim>>>,

      // RegisterWithObservers phase
      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterWithObservers<
                         Actions::ExportCoordinates<Dim>>,
                     Parallel::Actions::TerminatePhase>>,

      // Export phase
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Export,
                             tmpl::list<Actions::ExportCoordinates<Dim>>>>;

  // Set to a struct containining apply function that takes arguments
  // db::DataBox<DbTagsList>&&, options...) where options... are the
  // arguments read from the input file. Usually used to add options
  // for use during initialization. The simple_tags in the struct are
  // used as the tags that the apply function will add to the DataBox.
  // Cannot add any compute tags at this stage. A reasonable location
  // for this struct is insie the action that will do the initialization.
  using add_options_to_databox =
      typename Actions::InitializeElement<Dim>::AddOptionsToDataBox;

  // Effectively a constructor for the components
  // Called by Main parallel component when execution starts
  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const std::unique_ptr<DomainCreator<Dim, Frame::Inertial>>
          domain_creator) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& element_array =
        Parallel::get_parallel_component<ElementArray>(local_cache);

    auto domain = domain_creator->create_domain();
    for (const auto& block : domain.blocks()) {
      const auto initial_ref_levs =
          domain_creator->initial_refinement_levels()[block.id()];
      const std::vector<ElementId<Dim>> element_ids =
          initial_element_ids(block.id(), initial_ref_levs);
      int which_proc = 0;
      const int number_of_procs = Parallel::number_of_procs();
      for (size_t i = 0; i < element_ids.size(); ++i) {
        element_array(ElementIndex<Dim>(element_ids[i]))
            .insert(global_cache,
                    {domain_creator->initial_extents(),
                     domain_creator->create_domain()},
                    which_proc);
        which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
      }
    }
    element_array.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& element_array =
        Parallel::get_parallel_component<ElementArray>(local_cache);
    if (next_phase == Metavariables::Phase::CallArrayReduce) {
      Parallel::simple_action<ArrayReduce<Dim>>(element_array);
    }
    else {
        element_array.start_phase(next_phase);
    }
  }
};

template <size_t Dim>
struct Metavariables {
  // basic usage description
  static constexpr OptionString help{
      "Export the inertial coordinates of the Domain specified in the input "
      "file. The output can be used to compute initial data externally, for "
      "instance."};

  // (empty) list of OptionTags needed by metavariables
  using const_global_cache_tag_list = tmpl::list<>;
  // list of parallel components to be created
  using component_list = tmpl::list<SingletonParallelComponent
                                        <Dim, Metavariables>,
                                    ElementArray<Dim, Metavariables>,
                                    observers::Observer<Metavariables>,
                                    observers::ObserverWriter<Metavariables>>;
  using observed_reduction_data_tags = tmpl::list<>;

  // Phases to be executed, must contain at least Initialization and Exit
  enum class Phase { Initialization, RegisterWithObserver, Export,
                     CallArrayReduce, Exit };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Export;
      case Phase::Export:
        return Phase::CallArrayReduce;
      case Phase::CallArrayReduce:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR("Unknown type of phase.");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
