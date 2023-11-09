// // Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "Options/String.hpp"

namespace SpatialDiscretization::OptionTags {
/*!
 * \brief Group holding all the options for the spatial discretization of the
 * PDE system.
 *
 * For example, when using a discontinuous Galerkin scheme, options like the
 * basis function choice are specified within the SpatialDiscretizationGroup.
 * The boundary correction/numerical flux to use is another example of what
 * would be specified in the SpatialDiscretizationGroup. In the future,
 * specifying the domain decomposition should also be done in the spatial
 * discretization group.
 */
struct SpatialDiscretizationGroup {
  static std::string name() { return "SpatialDiscretization"; }
  static constexpr Options::String help{};
};
}  // namespace SpatialDiscretization::OptionTags
