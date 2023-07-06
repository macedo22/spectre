// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <vector>

#include "Options/String.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"

// this isn't the parallel component itself, but rather specifies the
// ingredients needed
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

struct Metavariables {
  // type list of parallel components
  using component_list = tmpl::list<PiEstimator<Metavariables>>;

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
