// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Elasticity {

namespace OptionTags {
template <size_t Dim>
struct ConstitutiveRelationPerBlock {
  static std::string name() noexcept { return "Material"; }
  using ConstRelPtr =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;
  using type =
      Options::Auto<std::variant<ConstRelPtr, std::vector<ConstRelPtr>,
                                 std::unordered_map<std::string, ConstRelPtr>>>;
  static constexpr Options::String help =
      "A constitutive relation in every block of the domain. Set to 'Auto' "
      "when solving an analytic solution to use the constitutive relation "
      "provided by the analytic solution.";
};
}  // namespace OptionTags

namespace Tags {
struct ConstitutiveRelationPerBlockBase : db::BaseTag {};

/// A constitutive relation in every block of the domain. Either constructed
/// from input-file options or retrieved from the analytic solution, if one
/// exists.
template <size_t Dim, typename BackgroundTag, typename AnalyticSolutionType>
struct ConstitutiveRelationPerBlock : db::SimpleTag,
                                      ConstitutiveRelationPerBlockBase {
  using ConstRelPtr =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;
  using type = std::vector<ConstRelPtr>;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>,
                             OptionTags::ConstitutiveRelationPerBlock<Dim>,
                             elliptic::OptionTags::Background<
                                 typename BackgroundTag::type::element_type>>;
  static type create_from_options(
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator,
      const std::optional<
          std::variant<ConstRelPtr, std::vector<ConstRelPtr>,
                       std::unordered_map<std::string, ConstRelPtr>>>&
          option_value,
      const typename BackgroundTag::type& background) noexcept {
    const auto block_names = domain_creator->block_names();
    const auto block_groups = domain_creator->block_groups();
    const size_t num_blocks = block_names.size();
    if (option_value.has_value()) {
      const domain::ExpandOverBlocks<ConstRelPtr> expand_over_blocks{
          block_names, block_groups};
      // lambda for printing IsotropicHomogeneous bulk_modulus and shear_modulus
      std::visit(
          [](auto&& arg) {
            // T is an alias to one of the std::variant options above:
            // (1) ConstRelPtr, which is an alias for
            // std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>
            // i.e. option_value's value is a ConstituiveRelation pointer
            // (2) std::vector<ConstRelPtr>
            // i.e. option_value's value is a vector of ConstitutiveRelation
            // pointers
            // (3) std::unordered_map<std::string, ConstRelPtr
            // i.e. option_value's value is a map from strings to
            // ConstitutiveRelation pointers
            using T = std::decay_t<decltype(arg)>;

            // Game plan of the code below:
            // (1) Determine which of the three types above option_value's
            // value is
            // (2) Attempt to downcast the ConstitutiveRelation part of the type
            // to an IsotropicHomogeneous type. If it succeeeds, then we know
            // option_value is (or holds a vector or map of)
            // IsotropicHomogeneous, as opposed to a CubicCrystal, which is
            // another derived type of ConstitutiveRelation, just like
            // IsotropicHomogeneous. In short, gotta make sure we are dealing
            // with an IsotropicHomogeneous thing.
            // (3) If we do have an IsotropicHomogeneous thing, then print
            // the bulk modulus/moduli and shear modulus/moduli. If
            // option_value's value is a vector or map, multiple are printed.

            // if option_value's value is a ConstitutiveRelation pointer
            if constexpr (std::is_same_v<T, ConstRelPtr>) {
              // attempt to downcast option_value's type from base type
              // ConstitutiveRelation to derived type IsotropicHomogeneous
              const auto derived_arg = dynamic_cast<
                  ConstitutiveRelations::IsotropicHomogeneous<Dim>*>(&(*arg));
              // if the cast was successful
              if (derived_arg) {
                // If the program enters this if block, then it means that
                // option_value's value is of type IsotropicHomogeneous
                // and derived_arg is an IsotropicHomogeneous pointer

                // call IsotropicHomogeneous bulk_modulus and shear_modulus
                // member functions to get the bulk modulus and shear modulus
                // and print the values
                Parallel::printf("bulk_modulus: %f, shear_modulus: %f\n\n",
                                 derived_arg->bulk_modulus(),
                                 derived_arg->shear_modulus());
              }
            }
            // otherwise, if option_value's value is a vector of
            // ConstitutiveRelation pointers
            else if constexpr (std::is_same_v<T, std::vector<ConstRelPtr>>) {
              // An iterator to iterate over the option_value vector. The next
              // line assigns it to refer to the first element of the vector.
              auto arg_it = arg.begin();
              // if the vector is not empty (i.e. has at least one element)
              if (arg_it != arg.end()) {
                // attempt to downcast the first ConstitutiveRelation pointer
                // element in the vector to an IsotropicHeomogeneous pointer
                const auto first_derived_arg_value = dynamic_cast<
                    ConstitutiveRelations::IsotropicHomogeneous<Dim>*>(
                    &(*(*arg_it)));
                // if the cast was successful
                if (first_derived_arg_value) {
                  // If the program enters this if block, then it means that
                  // option_value's first element is an IsotropicHomogeneous
                  // pointer. Note that no code below checks the other elements,
                  // it just assumes that if the first element is a pointer to
                  // an IsotropicHomogeneous object, the others are, too.

                  // call IsotropicHomogeneous bulk_modulus and shear_modulus
                  // member functions to get the bulk modulus and shear modulus
                  // of the IsotropicHomogenous object pointed to by the first
                  // pointer in the vector, then print their values
                  Parallel::printf(
                      "The modulus values are:\nbulk_modulus: %f, "
                      "shear_modulus: %f\n",
                      first_derived_arg_value->bulk_modulus(),
                      first_derived_arg_value->shear_modulus());
                  // move iterator to the next pointer in the vector
                  arg_it++;
                  // while we haven't reached the end of the vector
                  // (i.e. we still have more elements to print stuff about)
                  while (arg_it != arg.end()) {
                    // attempt to downcast this ConstitutiveRelation pointer
                    // element in the vector to an IsotropicHeomogeneous pointer
                    const auto derived_arg_value = dynamic_cast<
                        ConstitutiveRelations::IsotropicHomogeneous<Dim>*>(
                        &(*(*arg_it)));
                    // call IsotropicHomogeneous bulk_modulus and shear_modulus
                    // member functions to get the bulk modulus and shear
                    // modulus of the IsotropicHomogenous object pointed to by
                    // this pointer in the vector, then print their values
                    Parallel::printf("bulk_modulus: %f, shear_modulus: %f\n",
                                     derived_arg_value->bulk_modulus(),
                                     derived_arg_value->shear_modulus());
                    // move iterator to the next pointer in the vector
                    arg_it++;
                  }
                  Parallel::printf("\n");
                }
              }
            }
            // otherwise, if option_value's value is a map that maps strings to
            // ConstitutiveRelation pointers
            else if constexpr (std::is_same_v<
                                   T, std::unordered_map<std::string,
                                                         ConstRelPtr>>) {
              // Notes about this map:
              //
              // arg is a map where each element of the map is a key-value pair,
              // the keys being strings and the values that they map to being
              // ConstitutiveRelation pointers. If you have one element of the
              // map you can access the key with `first` and the value with
              // `second`, e.g. if `e` is one element (key-value pair) of our
              // map, then:
              //     std::string e_key = e.first;
              //     ConstRelPtr e_value = e.second;

              // An iterator to iterate over the option_value map. The next
              // line assigns it to refer to the first element (key-value pair)
              // of the map.
              auto arg_it = arg.begin();

              // if the map is not empty (i.e. has at least one element)
              if (arg_it != arg.end()) {
                // attempt to downcast the first ConstitutiveRelation pointer
                // value in the map to an IsotropicHomogeneous pointer
                auto first_derived_arg_value = dynamic_cast<
                    ConstitutiveRelations::IsotropicHomogeneous<Dim>*>(
                    &(*(arg_it->second)));
                // if the cast was successful
                if (first_derived_arg_value) {
                  // If the program enters this if block, then it means that
                  // option_value's first value is an IsotropicHomogeneous
                  // pointer. Note that no code below checks the other elements,
                  // it just assumes that if the first element's value is a
                  // pointer to an IsotropicHomogeneous object, the others are,
                  // too.

                  // call IsotropicHomogeneous bulk_modulus and shear_modulus
                  // member functions to get the bulk modulus and shear modulus
                  // of the IsotropicHomogenous object pointed to by the first
                  // pointer in the map, then print their values
                  Parallel::printf(
                      "The modulus values are:\nkey: %s, bulk_modulus: %f, "
                      "shear_modulus: %f\n",
                      arg_it->first, first_derived_arg_value->bulk_modulus(),
                      first_derived_arg_value->shear_modulus());
                  // move iterator to the next element in the map
                  arg_it++;
                  // while we haven't reached the end of the map
                  // (i.e. we still have more elements to print stuff about)
                  while (arg_it != arg.end()) {
                    // attempt to downcast this ConstitutiveRelation pointer
                    // in the map to an IsotropicHeomogeneous pointer
                    auto derived_arg_value = dynamic_cast<
                        ConstitutiveRelations::IsotropicHomogeneous<Dim>*>(
                        &(*(arg_it->second)));
                    // call IsotropicHomogeneous bulk_modulus and shear_modulus
                    // member functions to get the bulk modulus and shear
                    // modulus of the IsotropicHomogenous object pointed to by
                    // this pointer in the map, then print their values
                    Parallel::printf(
                        "key: %s, bulk_modulus: %f, shear_modulus: %f\n",
                        arg_it->first, derived_arg_value->bulk_modulus(),
                        derived_arg_value->shear_modulus());
                    // move iterator to the next element in the map
                    arg_it++;
                  }
                  Parallel::printf("\n");
                }
              }
            }
            // otherwise, option_value's value is none of the three type options
            // previously mentioned above
            else {
              // If you get here, it means there is a mistake somewhere in the
              // code, because it doesn't seem like it should be anything else

              // pls come up with a better error message lol
              ERROR("shouldn't get here");
            }
          },  // end lambda
          option_value.value());
      try {
        return std::visit(expand_over_blocks, *option_value);
      } catch (const std::exception& error) {
        ERROR("Invalid 'Material': " << error.what());
      }
    } else {
      const auto analytic_solution =
          dynamic_cast<const AnalyticSolutionType*>(background.get());
      if (analytic_solution == nullptr) {
        ERROR(
            "No analytic solution available that can provide a constitutive "
            "relation. Specify the 'Material' option.");
      } else {
        const auto& constitutive_relation =
            analytic_solution->constitutive_relation();
        type constitutive_relation_per_block{};
        constitutive_relation_per_block.reserve(num_blocks);
        for (size_t i = 0; i < num_blocks; ++i) {
          constitutive_relation_per_block.emplace_back(
              constitutive_relation.get_clone());
        }
        return constitutive_relation_per_block;
      }
    }
  }
};

/// References the constitutive relation for the element's block, which is
/// stored in the global cache
template <size_t Dim>
struct ConstitutiveRelationReference : ConstitutiveRelation<Dim>,
                                       db::ReferenceTag {
  using base = ConstitutiveRelation<Dim>;
  using argument_tags = tmpl::list<ConstitutiveRelationPerBlockBase,
                                   domain::Tags::Element<Dim>>;
  static const ConstitutiveRelations::ConstitutiveRelation<Dim>& get(
      const std::vector<
          std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>>&
          constitutive_relation_per_block,
      const Element<Dim>& element) noexcept {
    return *constitutive_relation_per_block.at(element.id().block_id());
  }
};
}  // namespace Tags

/// Actions related to solving Elasticity systems
namespace Actions {

/*!
 * \brief Initialize the constitutive relation describing properties of the
 * elastic material
 *
 * Every block in the domain can have a different constitutive relation,
 * allowing for composite materials. All constitutive relations are stored in
 * the global cache indexed by block, and elements reference their block's
 * constitutive relation in the DataBox. This means an element can retrieve the
 * local constitutive relation from the DataBox simply by requesting
 * `Elasticity::Tags::ConstitutiveRelation<Dim>`.
 */
template <size_t Dim, typename BackgroundTag, typename AnalyticSolutionType>
struct InitializeConstitutiveRelation {
 private:
 public:
  using const_global_cache_tags =
      tmpl::list<Elasticity::Tags::ConstitutiveRelationPerBlock<
          Dim, BackgroundTag, AnalyticSolutionType>>;
  using simple_tags = tmpl::list<>;
  using compute_tags =
      tmpl::list<Elasticity::Tags::ConstitutiveRelationReference<Dim>>;
  using initialization_tags = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace Elasticity
