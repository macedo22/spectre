// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"

namespace Parallel {
/// \cond
template <typename Metavariables>
struct ResourceInfo;
/// \endcond
namespace OptionTags {
/// \ingroup ParallelGroup
/// Options group for resource allocation
template <typename Metavariables>
struct ResourceInfo {
  using type = Parallel::ResourceInfo<Metavariables>;
  static Options::String help;
};
}  // namespace OptionTags

namespace Tags {
/// \ingroup ParallelGroup
/// Tag to retrieve the ResourceInfo.
template <typename Metavariables>
struct ResourceInfo : db::SimpleTag {
  using type = Parallel::ResourceInfo<Metavariables>;
};
}  // namespace Tags
}  // namespace Parallel
