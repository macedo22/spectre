// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
template <typename Metavariables>
struct get_main_option_list {
using component_list = typename Metavariables::component_list;
  using const_global_cache_tags = get_const_global_cache_tags<Metavariables>;
  using mutable_global_cache_tags =
      get_mutable_global_cache_tags<Metavariables>;
  template <typename ParallelComponent>
  using get_parallel_component_options =
      Parallel::get_option_tags<typename ParallelComponent::initialization_tags,
                                Metavariables>;
using type = tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
      Parallel::get_option_tags<const_global_cache_tags, Metavariables>,
      Parallel::get_option_tags<mutable_global_cache_tags, Metavariables>,
      tmpl::transform<component_list,
                      tmpl::bind<get_parallel_component_options, tmpl::_1>>>>>;
};
}  // namespace Parallel
