// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes and functions that handle parsing of input parameters.

#pragma once

#include <cerrno>
#include <cstring>
#include <exception>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <map>
#include <ostream>
#include <pup.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "Options/Options.hpp"
#include "Options/OptionsDetails.hpp"
#include "Options/Tags.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsMaplike.hpp"
#include "Utilities/TypeTraits/IsStdArray.hpp"
#include "Utilities/TypeTraits/IsStdArrayOfSize.hpp"

namespace Options {
namespace Options_detail {
template <typename T, typename Metavariables, typename Subgroup>
struct get_impl;
}  // namespace Options_detail

/// \ingroup OptionParsingGroup
/// \brief Class that handles parsing an input file
///
/// Options must be given YAML data to parse before output can be
/// extracted.  This can be done either from a file (parse_file
/// method), from a string (parse method), or, in the case of
/// recursive parsing, from an Option (parse method).  The options
/// can then be extracted using the get method.
///
/// \example
/// \snippet Test_Options.cpp options_example_scalar_struct
/// \snippet Test_Options.cpp options_example_scalar_parse
///
/// \see the \ref dev_guide_option_parsing tutorial
///
/// \tparam OptionList the list of option structs to parse
/// \tparam Group the option group with a group hierarchy
template <typename OptionList, typename Group = NoSuchType>
class Parser {
 private:
  /// All top-level options and top-level groups of options. Every option in
  /// `OptionList` is either in this list or in the hierarchy of one of the
  /// groups in this list.
  using tags_and_subgroups_list = tmpl::remove_duplicates<tmpl::transform<
      OptionList, Options_detail::find_subgroup<tmpl::_1, tmpl::pin<Group>>>>;

 public:
  Parser() = default;

  /// \param help_text an overall description of the options
  explicit Parser(std::string help_text);

  /// Parse a string to obtain options and their values.
  ///
  /// \param options the string holding the YAML formatted options
  void parse(std::string options);

  /// Parse an Option to obtain options and their values.
  void parse(const Option& options);

  /// Parse a file containing options
  ///
  /// \param file_name the path to the file to parse
  void parse_file(const std::string& file_name);

  /// Overlay the options from a string or file on top of the
  /// currently parsed options.
  ///
  /// Any tag included in the list passed as the template parameter
  /// can be overridden by a new parsed value.  Newly parsed options
  /// replace the previous values.  Any tags not appearing in the new
  /// input are left unchanged.
  /// @{
  template <typename OverlayOptions>
  void overlay(std::string options);

  template <typename OverlayOptions>
  void overlay_file(const std::string& file_name);
  /// @}

  /// Get the value of the specified option
  ///
  /// \tparam T the option to retrieve
  /// \return the value of the option
  template <typename T, typename Metavariables = NoSuchType>
  typename T::type get() const;

  /// Call a function with the specified options as arguments.
  ///
  /// \tparam TagList a typelist of options to pass
  /// \return the result of the function call
  template <typename TagList, typename Metavariables = NoSuchType, typename F>
  decltype(auto) apply(F&& func) const;

  /// Call a function with the typelist of parsed options (i.e., the
  /// supplied option list with the chosen branches of any
  /// Alternatives inlined) and the option values as arguments.
  ///
  /// \return the result of the function call.  This must have the
  /// same type for all valid sets of parsed arguments.
  template <typename Metavariables = NoSuchType, typename F>
  decltype(auto) apply_all(F&& func) const;

  /// Get the help string
  template <typename TagsAndSubgroups = tags_and_subgroups_list>
  std::string help() const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <typename, typename>
  friend class Parser;
  template <typename, typename, typename>
  friend struct Options_detail::get_impl;

  static_assert(tt::is_a<tmpl::list, OptionList>::value,
                "The OptionList template parameter to Options must be a "
                "tmpl::list<...>.");

  // All options that could be specified, including those that have
  // alternatives and are therefore not required.
  using all_possible_options = tmpl::remove_duplicates<
      typename Options_detail::flatten_alternatives<OptionList>::type>;

  static_assert(
      std::is_same_v<
          typename Options_detail::flatten_alternatives<OptionList>::type,
          OptionList> or
          tmpl::all<
              all_possible_options,
              std::is_same<tmpl::_1, Options_detail::find_subgroup<
                                         tmpl::_1, tmpl::pin<Group>>>>::value,
      "Option parser cannot handle Alternatives and options with groups "
      "simultaneously.");

  /// All top-level subgroups
  using subgroups = tmpl::list_difference<tags_and_subgroups_list, OptionList>;

  // The maximum length of an option label.
  static constexpr int max_label_size_ = 70;

  /// Parse a YAML node containing options
  void parse(const YAML::Node& node);

  /// Overlay data from a YAML node
  template <typename OverlayOptions>
  void overlay(const YAML::Node& node);

  /// Check that the size is not smaller than the lower bound
  ///
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_lower_bound_on_size(const typename T::type& t,
                                 const Context& context) const;

  /// Check that the size is not larger than the upper bound
  ///
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_upper_bound_on_size(const typename T::type& t,
                                 const Context& context) const;

  /// If the options has a lower bound, check it is satisfied.
  ///
  /// Note: Lower bounds are >=, not just >.
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_lower_bound(const typename T::type& t,
                         const Context& context) const;

  /// If the options has a upper bound, check it is satisfied.
  ///
  /// Note: Upper bounds are <=, not just <.
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_upper_bound(const typename T::type& t,
                         const Context& context) const;

  /// Get the help string for parsing errors
  template <typename TagsAndSubgroups = tags_and_subgroups_list>
  std::string parsing_help(const YAML::Node& options) const;

  /// Error message when failed to parse an input file.
  [[noreturn]] void parser_error(const YAML::Exception& e) const;

  template <typename ChosenOptions, typename RemainingOptions, typename F>
  auto call_with_chosen_alternatives_impl(F&& func,
                                          std::vector<size_t> choices) const;

  template <typename F>
  auto call_with_chosen_alternatives(F&& func) const {
    return call_with_chosen_alternatives_impl<tmpl::list<>, OptionList>(
        std::forward<F>(func), alternative_choices_);
  }

  std::string help_text_{};
  Context context_{};
  std::vector<std::string> input_source_{};
  std::unordered_map<std::string, YAML::Node> parsed_options_{};

  template <typename Subgroup>
  struct SubgroupParser {
    using type = Parser<Options_detail::options_in_group<OptionList, Subgroup>,
                        Subgroup>;
  };

  tuples::tagged_tuple_from_typelist<
      tmpl::transform<subgroups, tmpl::bind<SubgroupParser, tmpl::_1>>>
      subgroup_parsers_ =
          tmpl::as_pack<subgroups>([this](auto... subgroup_tags) {
            (void)this;  // gcc wants this for subgroup_parsers_
            return decltype(subgroup_parsers_)(
                tmpl::type_from<decltype(subgroup_tags)>::help...);
          });

  // The choices made for option alternatives in a depth-first order.
  // Starting from the front of the option list, when reaching the
  // first Alternatives object, replace it with the options in the nth
  // choice, where n is the *last* element of this vector.  Continue
  // processing from the start of those options, using the second to
  // last value here for the next choice, and so on.
  std::vector<size_t> alternative_choices_{};
};

template <typename OptionList, typename Group>
Parser<OptionList, Group>::Parser(std::string help_text)
    : help_text_(std::move(help_text)) {
  tmpl::for_each<all_possible_options>([](auto t) {
    using T = typename decltype(t)::type;
    const std::string label = pretty_type::name<T>();
    ASSERT(label.size() <= max_label_size_,
           "The option name " << label
                              << " is too long for nice formatting, "
                                 "please shorten the name to "
                              << max_label_size_ << " characters or fewer");
    ASSERT(std::strlen(T::help) > 0,
           "You must supply a help string of non-zero length for " << label);
  });
}

namespace Options_detail {
template <typename>
struct apply_helper;

template <typename... Tags>
struct apply_helper<tmpl::list<Tags...>> {
  template <typename Metavariables, typename Options, typename F>
  static decltype(auto) apply(const Options& opts, F&& func) {
    return func(opts.template get<Tags, Metavariables>()...);
  }
};
}  // namespace Options_detail

/// \cond
// Doxygen is confused by decltype(auto)
template <typename OptionList, typename Group>
template <typename TagList, typename Metavariables, typename F>
decltype(auto) Parser<OptionList, Group>::apply(F&& func) const {
  return Options_detail::apply_helper<TagList>::template apply<Metavariables>(
      *this, std::forward<F>(func));
}

template <typename OptionList, typename Group>
template <typename Metavariables, typename F>
decltype(auto) Parser<OptionList, Group>::apply_all(F&& func) const {
  return call_with_chosen_alternatives([this, &func](
                                           auto chosen_alternatives /*meta*/) {
    using ChosenAlternatives = decltype(chosen_alternatives);
    return this->apply<ChosenAlternatives, Metavariables>([&func](
                                                              auto&&... args) {
      return std::forward<F>(func)(ChosenAlternatives{}, std::move(args)...);
    });
  });
}
/// \endcond

template <typename OptionList, typename Group>
template <typename ChosenOptions, typename RemainingOptions, typename F>
auto Parser<OptionList, Group>::call_with_chosen_alternatives_impl(
    F&& func, std::vector<size_t> choices) const {
  if constexpr (std::is_same_v<RemainingOptions, tmpl::list<>>) {
    return std::forward<F>(func)(ChosenOptions{});
  } else {
    using next_option = tmpl::front<RemainingOptions>;
    using remaining_options = tmpl::pop_front<RemainingOptions>;

    if constexpr (not tt::is_a_v<Options::Alternatives, next_option>) {
      return call_with_chosen_alternatives_impl<
          tmpl::push_back<ChosenOptions, next_option>, remaining_options>(
          std::forward<F>(func), std::move(choices));
    } else {
      using Result =
          decltype(call_with_chosen_alternatives_impl<
                   ChosenOptions,
                   tmpl::append<tmpl::front<next_option>, remaining_options>>(
              std::forward<F>(func), std::move(choices)));

      const size_t choice = choices.back();
      choices.pop_back();

      Result result{};
      size_t alternative_number = 0;
      tmpl::for_each<next_option>([this, &alternative_number, &choice, &choices,
                                   &func, &result](auto alternative) {
        using Alternative = tmpl::type_from<decltype(alternative)>;
        if (choice == alternative_number++) {
          result = this->call_with_chosen_alternatives_impl<
              ChosenOptions, tmpl::append<Alternative, remaining_options>>(
              std::forward<F>(func), std::move(choices));
        }
      });
      return result;
    }
  }
}
}  // namespace Options
