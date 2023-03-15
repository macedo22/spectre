// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup ComputationalDomainGroup
struct ElementWeight {
  using type = ::domain::ElementWeight;
  static std::string name() { return "ElementWeight"; }
  static constexpr Options::String help =
      "Weighting scheme for assigning computational costs to Elements for "
      "distributing balanced compuational costs per processor";
};

/// \ingroup OptionTagsGroup
/// \ingroup ComputationalDomainGroup
template <typename ElementDistributionType>
struct ElementDistribution {
  using type = ElementDistributionType;
  static std::string name() { return "ElementDistribution"; }
  static constexpr Options::String help{"The time stepper"};
};
}  // namespace OptionTags

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The weighting scheme for assigning computational costs to `Element`s for
/// distributing balanced compuational costs per processor (see
/// `BlockZCurveProcDistribution`)
struct ElementWeight : db::SimpleTag {
  using type = ::domain::ElementWeight;
  using option_tags = tmpl::list<domain::OptionTags::ElementWeight>;

  static constexpr bool pass_metavariables = false;
  static ::domain::ElementWeight create_from_options(
      const ::domain::ElementWeight& element_weight);
};

template <typename ElementDistributionType>
struct ElementDistribution : db::SimpleTag {
  using type = ElementDistributionType;
  using option_tags = tmpl::list<
      ::domain::OptionTags::ElementDistribution<ElementDistributionType>>;

  static constexpr bool pass_metavariables = false;
  static ElementDistributionType create_from_options(
      const ElementDistributionType&
          element_distribution_type);
  ;
};
}  // namespace Tags
}  // namespace domain
