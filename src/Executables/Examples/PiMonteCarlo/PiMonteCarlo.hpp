// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <vector>

#include "Options/String.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/TMPL.hpp"

struct Metavariables {
  // type list of parallel components
  using component_list = tmpl::list<>;

  // phases of executable
  static constexpr std::array<Parallel::Phase, 3> default_phase_order{
      Parallel::Phase::Initialization, Parallel::Phase::Execute,
      Parallel::Phase::Exit};

  // "about" the executable
  static constexpr Options::String help{
      "Compute pi via monte carlo integration"};
};

// charm functions you want to turn on
static const std::vector<void (*)()> charm_init_node_funcs{};
static const std::vector<void (*)()> charm_init_proc_funcs{};
