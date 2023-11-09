// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/SpatialDiscretization/OptionTags.hpp"
#include "Options/String.hpp"

namespace dg::OptionTags {
/*!
 * \brief Group holding options for controlling the DG discretization.
 *
 * For example, this would hold whether to use the strong or weak form, what
 * boundary correction/numerical flux to use, and which quadrature rule to use.
 *
 * \note The `DiscontinuousGalerkinGroup` is a subgroup of
 * `SpatialDiscretization::OptionTags::SpatialDiscretizationGroup`.
 */
struct DiscontinuousGalerkinGroup {
  static std::string name() { return "DiscontinuousGalerkin"; }
  static constexpr Options::String help{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"};
  using group = SpatialDiscretization::OptionTags::SpatialDiscretizationGroup;
};
}  // namespace dg::OptionTags
