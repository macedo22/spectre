// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CCZ4/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/Options.hpp"

class DataVector;

namespace CCZ4 {
namespace Tags {
/*!
 * \brief Compute item for the conformal factor \f$\phi\f$ used by the CCZ4
 * formulation of Einstein's equations.
 */
template <typename DataType>
struct Phi : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief Compute item for the square of the conformal factor, \f$\phi^2\f$,
 * used by the CCZ4 formulation of Einstein's equations.
 */
template <typename DataType>
struct PhiSquared : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief Compute item for the conformal spatial metric
 * \f$\widetilde{\gamma}_{ij}\f$ used by the CCZ4 formulation of Einstein's
 * equations.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct ConfSpatialMetric : db::SimpleTag {
  using type = tnsr::ii<DataType, SpatialDim, Frame>;
};
}  // namespace Tags

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * Groups option tags related to the GeneralizedHarmonic evolution system.
 */
struct Group {
  static std::string name() noexcept { return "CCZ4"; }
  static constexpr Options::String help{
      "Options for the CCZ4 evolution system"};
  using group = evolution::OptionTags::SystemGroup;
};
}  // namespace OptionTags
}  // namespace CCZ4
