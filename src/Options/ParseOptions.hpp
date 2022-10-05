// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes and functions that handle parsing of input parameters.

#pragma once

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <ostream>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "Options/Options.hpp"
#include "Options/OptionsDetails.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
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

  #if defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wundefined-internal"
  #endif  // defined(__clang__)
  /// Get the value of the specified option
  ///
  /// \tparam T the option to retrieve
  /// \return the value of the option
  template <typename T, typename Metavariables = NoSuchType>
  typename T::type __attribute__((used)) get() const;
  #if defined(__clang__)
  #pragma GCC diagnostic pop
  #endif  // defined(__clang__)

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

  /// \cond
  // Doxygen is confused by decltype(auto)
  // Parse a YAML node containing options
  void __attribute__((used)) parse(const YAML::Node& node);
  /// \endcond

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

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse(std::string options) {
  context_.append("In string");
  input_source_.push_back(std::move(options));
  try {
    parse(YAML::Load(input_source_.back()));
  } catch (const YAML::Exception& e) {
    parser_error(e);
  }
}

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse(const Option& options) {
  context_ = options.context();
  parse(options.node());
}

namespace Options_detail {
inline std::ifstream open_file(const std::string& file_name) {
  errno = 0;
  std::ifstream input(file_name);
  if (not input) {
    // There is no standard way to get an error message from an
    // fstream, but this works on many implementations.
    ERROR("Could not open " << file_name << ": " << strerror(errno));
  }
  return input;
}
}  // namespace Options_detail

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse_file(const std::string& file_name) {
  context_.append("In " + file_name);
  auto input = Options_detail::open_file(file_name);
  input_source_.push_back(std::string(std::istreambuf_iterator(input), {}));
  try {
    parse(YAML::Load(input_source_.back()));
  } catch (const YAML::Exception& e) {
    parser_error(e);
  }
}

namespace Options_detail {
// Attempts to match the given_options against OptionList, choosing
// between any alternatives to maximize the number of matches.
// Returns the highest number of matches and the choices required to
// obtain that number.  (See the description of
// Parser::alternative_choices_ for the format of the choices.)
//
// In the case of multiple equally good matches, the choice between
// them will be given as std::numeric_limits<size_t>::max().  This may
// cause a failure later or may be ignored if a better choice is
// found.
//
// This does not handle groups correctly, but we disallow alternatives
// when groups are present so there is only one possible choice that
// this function could make.
template <typename OptionList>
std::pair<int, std::vector<size_t>> choose_alternatives(
    const std::unordered_set<std::string>& given_options) {
  int num_matched = 0;
  std::vector<size_t> alternative_choices{};
  tmpl::for_each<OptionList>([&alternative_choices, &num_matched,
                              &given_options](auto opt) {
    using Opt = tmpl::type_from<decltype(opt)>;
    if constexpr (not tt::is_a_v<Options::Alternatives, Opt>) {
      if (given_options.count(pretty_type::name<Opt>()) == 1) {
        ++num_matched;
      }
    } else {
      int most_matches = 0;
      std::vector<size_t> best_alternatives{std::numeric_limits<size_t>::max()};

      size_t alternative_number = 0;
      tmpl::for_each<Opt>([&alternative_number, &best_alternatives,
                           &most_matches, &given_options](auto alternative) {
        using Alternative = tmpl::type_from<decltype(alternative)>;
        auto alternative_match =
            choose_alternatives<Alternative>(given_options);
        if (alternative_match.first > most_matches) {
          most_matches = alternative_match.first;
          alternative_match.second.push_back(alternative_number);
          best_alternatives = std::move(alternative_match.second);
        } else if (alternative_match.first == most_matches) {
          // Two equally good matches
          best_alternatives.clear();
          best_alternatives.push_back(std::numeric_limits<size_t>::max());
        }
        ++alternative_number;
      });
      num_matched += most_matches;
      alternative_choices.insert(alternative_choices.begin(),
                                 best_alternatives.begin(),
                                 best_alternatives.end());
    }
  });
  return {num_matched, std::move(alternative_choices)};
}
}  // namespace Options_detail

template <typename OptionList, typename Group>
template <typename OverlayOptions>
void Parser<OptionList, Group>::overlay(std::string options) {
  context_ = Context{};
  context_.append("In string");
  input_source_.push_back(std::move(options));
  try {
    overlay<OverlayOptions>(YAML::Load(input_source_.back()));
  } catch (const YAML::Exception& e) {
    parser_error(e);
  }
}

template <typename OptionList, typename Group>
template <typename OverlayOptions>
void Parser<OptionList, Group>::overlay_file(const std::string& file_name) {
  context_ = Context{};
  context_.append("In " + file_name);
  auto input = Options_detail::open_file(file_name);
  input_source_.push_back(std::string(std::istreambuf_iterator(input), {}));
  try {
    overlay<OverlayOptions>(YAML::Load(input_source_.back()));
  } catch (const YAML::Exception& e) {
    parser_error(e);
  }
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
template <typename TagsAndSubgroups>
std::string Parser<OptionList, Group>::help() const {
  std::ostringstream ss;
  ss << "\n==== Description of expected options:\n" << help_text_;
  if (tmpl::size<TagsAndSubgroups>::value > 0) {
    ss << "\n\nOptions:\n"
       << tmpl::for_each<TagsAndSubgroups>(Options_detail::print<OptionList>{})
              .value;
  } else {
    ss << "\n\n<No options>\n";
  }
  return ss.str();
}

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::pup(PUP::er& p) {
  static_assert(std::is_same_v<Group, NoSuchType>,
                "Inner parsers should be recreated by the root parser, not "
                "serialized.");
  // We reconstruct most of the state when deserializing, rather than
  // trying to package it.
  p | help_text_;
  p | input_source_;
  if (p.isUnpacking() and not input_source_.empty()) {
    // input_source_ is populated by the `parse` and `overlay` calls
    // below, so we have to clear out the old values before calling
    // them.
    auto received_source = std::move(input_source_);
    input_source_.clear();
    parse(std::move(received_source[0]));
    for (size_t i = 1; i < received_source.size(); ++i) {
      overlay<OptionList>(std::move(received_source[i]));
    }
  }
}

template <typename OptionList, typename Group>
template <typename OverlayOptions>
void Parser<OptionList, Group>::overlay(const YAML::Node& node) {
  // This could be relaxed to allow mandatory options in a list with
  // alternatives to be overlaid (or even any options in the chosen
  // alternative), but overlaying is only done at top level and we
  // don't use alternatives there (because they conflict with groups,
  // which we do use).
  static_assert(
      std::is_same_v<
          typename Options_detail::flatten_alternatives<OptionList>::type,
          OptionList>,
      "Cannot overlay options when using alternatives.");
  static_assert(
      std::is_same_v<tmpl::list_difference<OverlayOptions, OptionList>,
                     tmpl::list<>>,
      "Can only overlay options that were originally parsed.");

  using overlayable_tags_and_subgroups_list =
      tmpl::remove_duplicates<tmpl::transform<
          OverlayOptions,
          Options_detail::find_subgroup<tmpl::_1, tmpl::pin<Group>>>>;

  if (not(node.IsMap() or node.IsNull())) {
    PARSE_ERROR(context_, "'" << node << "' does not look like options.\n"
                              << help<overlayable_tags_and_subgroups_list>());
  }

  std::unordered_set<std::string> overlaid_options{};
  overlaid_options.reserve(node.size());

  for (const auto& name_and_value : node) {
    const auto& name = name_and_value.first.as<std::string>();
    const auto& value = name_and_value.second;
    auto context = context_;
    context.line = name_and_value.first.Mark().line;
    context.column = name_and_value.first.Mark().column;

    if (tmpl::as_pack<tags_and_subgroups_list>([&name](auto... opts) {
          return (
              (name != pretty_type::name<tmpl::type_from<decltype(opts)>>()) and
              ...);
        })) {
      PARSE_ERROR(context,
                  "Option '"
                      << name << "' is not a valid option.\n"
                      << parsing_help<overlayable_tags_and_subgroups_list>(
                             node));
    }

    if (tmpl::as_pack<overlayable_tags_and_subgroups_list>(
            [&name](auto... opts) {
              return ((name !=
                       pretty_type::name<tmpl::type_from<decltype(opts)>>()) and
                      ...);
            })) {
      PARSE_ERROR(context,
                  "Option '"
                      << name << "' is not overlayable.\n"
                      << parsing_help<overlayable_tags_and_subgroups_list>(
                             node));
    }

    // Check for duplicate key
    if (0 != overlaid_options.count(name)) {
      PARSE_ERROR(context,
                  "Option '"
                      << name << "' specified twice.\n"
                      << parsing_help<overlayable_tags_and_subgroups_list>(
                             node));
    }

    overlaid_options.insert(name);
    parsed_options_.at(name) = value;
  }

  tmpl::for_each<subgroups>([this, &overlaid_options](auto subgroup_v) {
    using subgroup = tmpl::type_from<decltype(subgroup_v)>;
    if (overlaid_options.count(pretty_type::name<subgroup>()) == 1) {
      auto& subgroup_parser =
          tuples::get<SubgroupParser<subgroup>>(subgroup_parsers_);
      subgroup_parser.template overlay<
          Options_detail::options_in_group<OverlayOptions, subgroup>>(
          parsed_options_.find(pretty_type::name<subgroup>())->second);
    }
  });
}

template <typename OptionList, typename Group>
template <typename T>
void Parser<OptionList, Group>::check_lower_bound_on_size(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_lower_bound_on_size<T>::value) {
    static_assert(std::is_same_v<decltype(T::lower_bound_on_size()), size_t>,
                  "lower_bound_on_size() is not a size_t.");
    if (t.size() < T::lower_bound_on_size()) {
      PARSE_ERROR(context, "Value must have at least "
                               << T::lower_bound_on_size() << " entries, but "
                               << t.size() << " were given.\n"
                               << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename T>
void Parser<OptionList, Group>::check_upper_bound_on_size(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_upper_bound_on_size<T>::value) {
    static_assert(std::is_same_v<decltype(T::upper_bound_on_size()), size_t>,
                  "upper_bound_on_size() is not a size_t.");
    if (t.size() > T::upper_bound_on_size()) {
      PARSE_ERROR(context, "Value must have at most "
                               << T::upper_bound_on_size() << " entries, but "
                               << t.size() << " were given.\n"
                               << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename T>
inline void Parser<OptionList, Group>::check_lower_bound(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_lower_bound<T>::value) {
    static_assert(std::is_same_v<decltype(T::lower_bound()), typename T::type>,
                  "Lower bound is not of the same type as the option.");
    static_assert(not std::is_same_v<typename T::type, bool>,
                  "Cannot set a lower bound for a bool.");
    if (t < T::lower_bound()) {
      PARSE_ERROR(context, "Value " << (MakeString{} << t)
                                    << " is below the lower bound of "
                                    << (MakeString{} << T::lower_bound())
                                    << ".\n" << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename T>
inline void Parser<OptionList, Group>::check_upper_bound(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_upper_bound<T>::value) {
    static_assert(std::is_same_v<decltype(T::upper_bound()), typename T::type>,
                  "Upper bound is not of the same type as the option.");
    static_assert(not std::is_same_v<typename T::type, bool>,
                  "Cannot set an upper bound for a bool.");
    if (t > T::upper_bound()) {
      PARSE_ERROR(context, "Value " << (MakeString{} << t)
                                    << " is above the upper bound of "
                                    << (MakeString{} << T::upper_bound())
                                    << ".\n" << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename TagsAndSubgroups>
std::string Parser<OptionList, Group>::parsing_help(
    const YAML::Node& options) const {
  std::ostringstream os;
  // At top level this would dump the entire input file, which is very
  // verbose and not very informative.  At lower levels the result
  // should be much shorter and may actually give useful context for
  // what part of the file is being parsed.
  if (not context_.top_level) {
    os << "\n==== Parsing the option string:\n" << options << "\n";
  }
  os << help<TagsAndSubgroups>();
  return os.str();
}

template <typename OptionList, typename Group>
[[noreturn]] void Parser<OptionList, Group>::parser_error(
    const YAML::Exception& e) const {
  auto context = context_;
  context.line = e.mark.line;
  context.column = e.mark.column;
  // Inline the top_level branch of PARSE_ERROR to avoid warning that
  // the other branch would call terminate.  (Parser errors can only
  // be generated at top level.)
  ERROR(
      "\n"
      << context
      << "Unable to correctly parse the input file because of a syntax error.\n"
         "This is often due to placing a suboption on the same line as an "
         "option, e.g.:\nDomainCreator: CreateInterval:\n  IsPeriodicIn: "
         "[false]\n\nShould be:\nDomainCreator:\n  CreateInterval:\n    "
         "IsPeriodicIn: [true]\n\nSee an example input file for help.");
}

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
