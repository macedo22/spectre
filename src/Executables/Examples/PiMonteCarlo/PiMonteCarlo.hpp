// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_set>
#include <vector>

#include "Options/String.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/DistributedObject.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

template <typename Metavars>
struct PiEstimator;

template <typename Metavars>
struct DartThrower;

// this isn't the parallel component itself, but rather specifies the
// ingredients needed
//
// this is a singleton so it lives on one core
template <typename Metavars>
struct PiEstimator {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavars;
  // type list of Actions that the code will do over and over in some phase
  // until it hits some stop condition, where first arg to PhaseActions is the
  // phase and second arg is the list of actions (tasks) to do
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<>>>;
  using simple_tags_from_options = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& global_cache);
};

template <typename Metavars>
void PiEstimator<Metavars>::execute_next_phase(
    const Parallel::Phase next_phase,
    const Parallel::CProxy_GlobalCache<Metavars>& global_cache) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  // start_phase starts the phase dependent action list?
  Parallel::get_parallel_component<PiEstimator<Metavars>>(local_cache)
      .start_phase(next_phase);
}

// this is an array so it lives on some number of cores
template <typename Metavars>
struct DartThrower {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavars;
  // type list of Actions that the code will do over and over in some phase
  // until it hits some stop condition, where first arg to PhaseActions is the
  // phase and second arg is the list of actions (tasks) to do
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<>>>;
  using simple_tags_from_options = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;

  using array_index = int;  // size_t would be nice but Charm uses int

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& global_cache);

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavars>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_options,
      const std::unordered_set<size_t>& procs_to_ignore = {});
};

template <typename Metavars>
void DartThrower<Metavars>::execute_next_phase(
    const Parallel::Phase next_phase,
    const Parallel::CProxy_GlobalCache<Metavars>& global_cache) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  // start_phase starts the phase dependent action list?
  Parallel::get_parallel_component<DartThrower<Metavars>>(local_cache)
      .start_phase(next_phase);
}

template <typename Metavars>
void DartThrower<Metavars>::allocate_array(
    Parallel::CProxy_GlobalCache<Metavars>& global_cache,
    const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
        initialization_options,
    const std::unordered_set<size_t>& procs_to_ignore) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  auto& array_proxy =
      Parallel::get_parallel_component<DartThrower<Metavars>>(local_cache);

  size_t which_proc = 0;  // the processor we're putting the next element on
  const size_t num_procs = Parallel::number_of_procs<size_t>(local_cache);
  const size_t number_of_elements = num_procs;

  // round robin assignment of elements to procs
  for (size_t i = 0; i < number_of_elements; i++) {
    // skip any processor to ignore
    while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
      which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
    }
    // allocate element i
    array_proxy[i].insert(global_cache, initialization_options, which_proc);
    which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
  }
  array_proxy.doneInserting();
}

struct Metavariables {
  // type list of parallel components
  using component_list =
      tmpl::list<PiEstimator<Metavariables>, DartThrower<Metavariables>>;

  // phases of executable
  static constexpr std::array<Parallel::Phase, 3> default_phase_order{
      Parallel::Phase::Initialization, Parallel::Phase::Execute,
      Parallel::Phase::Exit};

  // "about" the executable
  static constexpr Options::String help{
      "Compute pi via monte carlo integration"};

  void pup(PUP::er& /*p*/) {}
};

// charm functions you want to turn on
static const std::vector<void (*)()> charm_init_node_funcs{};
static const std::vector<void (*)()> charm_init_proc_funcs{};
