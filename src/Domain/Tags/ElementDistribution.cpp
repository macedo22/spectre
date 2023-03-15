// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Tags/ElementDistribution.hpp"

#include <cstddef>
#include <memory>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::Tags {
template <size_t Dim>
std::vector<std::array<size_t, Dim>> InitialExtents<Dim>::create_from_options(
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
  return domain_creator->initial_extents();
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                          \
  template std::vector<std::array<size_t, DIM(data)>> \
  InitialExtents<DIM(data)>::create_from_options(     \
      const std::unique_ptr<::DomainCreator<DIM(data)>>& domain_creator);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

::domain::ElementWeight ElementWeight::create_from_options(
    const ::domain::ElementWeight& element_weight) {
  return element_weight;
}

template <typename ElementDistributionType>
struct ElementDistribution : ElementDistribution<>, db::SimpleTag {
  using type = std::unique_ptr<ElementDistributionType>;
  using option_tags = tmpl::list<
      ::domain::OptionTags::ElementDistribution<ElementDistributionType>>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<ElementDistributionType> create_from_options(
      const std::unique_ptr<ElementDistributionType>&
          element_distribution_type);
  ;
};

::domain::ElementWeight ElementWeight::create_from_options(
    const ::domain::ElementWeight& element_weight) {
  return element_weight;
}
}  // namespace domain::Tags
