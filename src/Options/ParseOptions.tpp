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
#include "Options/ParseOptions.hpp"
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
// // Defining methods as inline in a different header from the class
// // definition is somewhat strange.  It is done here to minimize the
// // amount of code in the frequently-included Options.hpp file.  The
// // only external consumers of Option should be create_from_yaml
// // specializations, and they should only be instantiated by code in
// // this file.  (Or explicitly instantiated in cpp files, which can
// // include this file.)

// // clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
// // NOLINTNEXTLINE(performance-unnecessary-value-param)
// inline Option::Option(YAML::Node node, Context context)
//     : node_(std::make_unique<YAML::Node>(std::move(node))),
//       context_(std::move(context)) {  // NOLINT
//   context_.line = node.Mark().line;
//   context_.column = node.Mark().column;
// }

// inline Option::Option(Context context)
//     : node_(std::make_unique<YAML::Node>()), context_(std::move(context)) {}

// inline const YAML::Node& Option::node() const { return *node_; }
// inline const Context& Option::context() const { return context_; }

// /// Append a line to the contained context.
// inline void Option::append_context(const std::string& context) {
//   context_.append(context);
// }

// // NOLINTNEXTLINE(performance-unnecessary-value-param)
// inline void Option::set_node(YAML::Node node) {
//   // clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
//   *node_ = std::move(node);  // NOLINT
//   context_.line = node_->Mark().line;
//   context_.column = node_->Mark().column;
// }

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 8
template <typename T, typename Metavariables>
T Option::parse_as() const {
  try {
    // yaml-cpp's `as` method won't parse empty nodes, so we need to
    // inline a bit of its logic.
    Options_detail::wrap_create_types<T, Metavariables> result{};
    if (YAML::convert<decltype(result)>::decode(node(), result)) {
      return Options_detail::unwrap_create_types(std::move(result));
    }
    // clang-tidy: thrown exception is not nothrow copy constructible
    throw YAML::BadConversion(node().Mark());  // NOLINT
  } catch (const YAML::BadConversion& e) {
    // This happens when trying to parse an empty value as a container
    // with no entries.
    if ((tt::is_a_v<std::vector, T> or tt::is_std_array_of_size_v<0, T> or
         tt::is_maplike_v<T>) and node().IsNull()) {
      return T{};
    }
    Context error_context = context();
    error_context.line = e.mark.line;
    error_context.column = e.mark.column;
    std::ostringstream ss;
    ss << "Failed to convert value to type "
       << Options_detail::yaml_type<T>::value() << ":";

    const std::string value_text = YAML::Dump(node());
    if (value_text.find('\n') == std::string::npos) {
      ss << " " << value_text;
    } else {
      // Indent each line of the value by two spaces and start on a new line
      ss << "\n  ";
      for (char c : value_text) {
        ss << c;
        if (c == '\n') {
          ss << "  ";
        }
      }
    }

    if (tt::is_a_v<std::vector, T> or tt::is_std_array_v<T>) {
      ss << "\n\nNote: For sequences this can happen because the length of the "
            "sequence specified\nin the input file is not equal to the length "
            "expected by the code. Sequences in\nfiles can be denoted either "
            "as a bracket enclosed list ([foo, bar]) or with each\nentry on a "
            "separate line, indented and preceeded by a dash (  - foo).";
    }
    PARSE_ERROR(error_context, ss.str());
  } catch (const Options_detail::propagate_context& e) {
    Context error_context = context();
    // Avoid line numbers in the middle of the trace
    error_context.line = -1;
    error_context.column = -1;
    PARSE_ERROR(error_context, e.message());
  } catch (std::exception& e) {
    ERROR("Unexpected exception: " << e.what());
  }
}
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 8
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 8

// template <typename OptionList, typename Group>
// void Parser<OptionList, Group>::parse(std::string options) {
//   context_.append("In string");
//   input_source_.push_back(std::move(options));
//   try {
//     parse(YAML::Load(input_source_.back()));
//   } catch (const YAML::Exception& e) {
//     parser_error(e);
//   }
// }

// template <typename OptionList, typename Group>
// void Parser<OptionList, Group>::parse(const Option& options) {
//   context_ = options.context();
//   parse(options.node());
// }

// template <typename OptionList, typename Group>
// void Parser<OptionList, Group>::parse_file(const std::string& file_name) {
//   context_.append("In " + file_name);
//   auto input = Options_detail::open_file(file_name);
//   input_source_.push_back(std::string(std::istreambuf_iterator(input), {}));
//   try {
//     parse(YAML::Load(input_source_.back()));
//   } catch (const YAML::Exception& e) {
//     parser_error(e);
//   }
// }

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

namespace Options_detail {
template <typename Tag, typename Metavariables, typename Subgroup>
struct get_impl {
  template <typename OptionList, typename Group>
  static typename Tag::type apply(const Parser<OptionList, Group>& opts) {
    static_assert(
        tmpl::list_contains_v<OptionList, Tag>,
        "Could not find requested option in the list of options provided. Did "
        "you forget to add the option tag to the OptionList?");
    return tuples::get<typename Parser<
        OptionList, Group>::template SubgroupParser<Subgroup>>(
               opts.subgroup_parsers_)
        .template get<Tag, Metavariables>();
  }
};

template <typename Tag, typename Metavariables>
struct get_impl<Tag, Metavariables, Tag> {
  template <typename OptionList, typename Group>
  static typename Tag::type apply(const Parser<OptionList, Group>& opts) {
    static_assert(
        tmpl::list_contains_v<
            typename Parser<OptionList, Group>::all_possible_options, Tag>,
        "Could not find requested option in the list of options provided. Did "
        "you forget to add the option tag to the OptionList?");
    const std::string label = pretty_type::name<Tag>();

    const auto supplied_option = opts.parsed_options_.find(label);
    ASSERT(supplied_option != opts.parsed_options_.end(),
           "Requested option from alternative that was not supplied.");
    Option option(supplied_option->second, opts.context_);
    option.append_context("While parsing option " + label);

    auto t = option.parse_as<typename Tag::type, Metavariables>();

    if constexpr (Options_detail::has_suggested<Tag>::value) {
      static_assert(
          std::is_same_v<decltype(Tag::suggested_value()), typename Tag::type>,
          "Suggested value is not of the same type as the option.");

      // This can be easily relaxed, but using it would require
      // writing comparison operators for abstract base classes.  If
      // someone wants this enough to go though the effort of doing
      // that, it would just require comparing the dereferenced
      // pointers below to decide whether the suggestion was followed.
      static_assert(not tt::is_a_v<std::unique_ptr, typename Tag::type>,
                    "Suggestions are not supported for pointer types.");

      const auto suggested_value = Tag::suggested_value();
      {
        Context context;
        context.append("Checking SUGGESTED value for " +
                       pretty_type::name<Tag>());
        opts.template check_lower_bound_on_size<Tag>(suggested_value, context);
        opts.template check_upper_bound_on_size<Tag>(suggested_value, context);
        opts.template check_lower_bound<Tag>(suggested_value, context);
        opts.template check_upper_bound<Tag>(suggested_value, context);
      }

      if (t != suggested_value) {
        Parallel::printf_error(
            "%s, line %d:\n  Specified: %s\n  Suggested: %s\n",
            label, option.context().line + 1,
            (MakeString{} << std::boolalpha << t),
            (MakeString{} << std::boolalpha << suggested_value));
      }
    }

    opts.template check_lower_bound_on_size<Tag>(t, option.context());
    opts.template check_upper_bound_on_size<Tag>(t, option.context());
    opts.template check_lower_bound<Tag>(t, option.context());
    opts.template check_upper_bound<Tag>(t, option.context());
    return t;
  }
};

template <typename Metavariables>
struct get_impl<Tags::InputSource, Metavariables, Tags::InputSource> {
  template <typename OptionList, typename Group>
  static Tags::InputSource::type apply(const Parser<OptionList, Group>& opts) {
    return opts.input_source_;
  }
};
}  // namespace Options_detail

// template <typename...T>
// struct td;

template <typename OptionList, typename Group>
template <typename Tag, typename Metavariables>
typename Tag::type __attribute__((used)) Parser<OptionList, Group>::get() const {
  // td<Metavariables, OptionList, Group, Tag> idk{};
  return Options_detail::get_impl<
      Tag, Metavariables,
      typename Options_detail::find_subgroup<Tag, Group>::type>::apply(*this);
}

// template <typename OptionList, typename Group>
// template <typename TagsAndSubgroups>
// std::string Parser<OptionList, Group>::help() const {
//   std::ostringstream ss;
//   ss << "\n==== Description of expected options:\n" << help_text_;
//   if (tmpl::size<TagsAndSubgroups>::value > 0) {
//     ss << "\n\nOptions:\n"
//        << tmpl::for_each<TagsAndSubgroups>(Options_detail::print<OptionList>{})
//               .value;
//   } else {
//     ss << "\n\n<No options>\n";
//   }
//   return ss.str();
// }

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
void Parser<OptionList, Group>::parse(const YAML::Node& node) {
  if (not(node.IsMap() or node.IsNull())) {
    PARSE_ERROR(context_, "'" << node << "' does not look like options.\n"
                              << help());
  }

  std::unordered_set<std::string> given_options{};
  for (const auto& name_and_value : node) {
    given_options.insert(name_and_value.first.as<std::string>());
  }

  alternative_choices_ =
      Options_detail::choose_alternatives<OptionList>(given_options).second;
  if (alg::any_of(alternative_choices_, [](const size_t x) {
        return x == std::numeric_limits<size_t>::max();
      })) {
    PARSE_ERROR(context_, "Cannot decide between alternative options.\n"
                              << parsing_help(node));
  }

  auto valid_names = call_with_chosen_alternatives([](auto option_list_v) {
    using option_list = decltype(option_list_v);
    using top_level_options_and_groups =
        tmpl::remove_duplicates<tmpl::transform<
            option_list,
            Options_detail::find_subgroup<tmpl::_1, tmpl::pin<Group>>>>;
    // Use an ordered container so the missing options are reported in
    // the order they are given in the help string.
    std::vector<std::string> result;
    result.reserve(tmpl::size<top_level_options_and_groups>{});
    tmpl::for_each<top_level_options_and_groups>([&result](auto opt) {
      using Opt = tmpl::type_from<decltype(opt)>;
      const std::string label = pretty_type::name<Opt>();
      ASSERT(alg::find(result, label) == result.end(),
             "Duplicate option name: " << label);
      result.push_back(label);
    });
    return result;
  });

  for (const auto& name_and_value : node) {
    const auto& name = name_and_value.first.as<std::string>();
    const auto& value = name_and_value.second;
    auto context = context_;
    context.line = name_and_value.first.Mark().line;
    context.column = name_and_value.first.Mark().column;

    // Check for duplicate key
    if (0 != parsed_options_.count(name)) {
      PARSE_ERROR(context, "Option '" << name << "' specified twice.\n"
                                      << parsing_help(node));
    }

    // Check for invalid key
    const auto name_it = alg::find(valid_names, name);
    if (name_it == valid_names.end()) {
      tmpl::for_each<all_possible_options>([this, &context, &name,
                                            &node](auto tag) {
        using Tag = tmpl::type_from<decltype(tag)>;
        if (name == pretty_type::name<Tag>()) {
          PARSE_ERROR(context,
                      "Option '"
                          << name
                          << "' is unused because of other provided options.\n"
                          << parsing_help(node));
        }
      });
      PARSE_ERROR(context, "Option '" << name << "' is not a valid option.\n"
                                      << parsing_help(node));
    }

    parsed_options_.emplace(name, value);
    valid_names.erase(name_it);
  }

  if (not valid_names.empty()) {
    PARSE_ERROR(context_, "You did not specify the option"
                << (valid_names.size() == 1 ? " " : "s ")
                << (MakeString{} << valid_names) << "\n" << parsing_help(node));
  }

  tmpl::for_each<subgroups>([this](auto subgroup_v) {
    using subgroup = tmpl::type_from<decltype(subgroup_v)>;
    auto& subgroup_parser =
        tuples::get<SubgroupParser<subgroup>>(subgroup_parsers_);
    subgroup_parser.context_ = context_;
    subgroup_parser.context_.append("In group " +
                                    pretty_type::name<subgroup>());
    subgroup_parser.parse(
        parsed_options_.find(pretty_type::name<subgroup>())->second);
  });

  // Any actual warnings will be printed by later calls to get or
  // apply, but it is not clear how to determine in those functions
  // whether this message should be printed.
  if (std::is_same_v<Group, NoSuchType> and context_.top_level) {
    Parallel::printf_error(
        "The following options differ from their suggested values:\n");
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
                  "Option '" << name << "' is not a valid option.\n"
                  << parsing_help<overlayable_tags_and_subgroups_list>(node));
    }

    if (tmpl::as_pack<overlayable_tags_and_subgroups_list>(
            [&name](auto... opts) {
              return ((name !=
                       pretty_type::name<tmpl::type_from<decltype(opts)>>()) and
                      ...);
            })) {
      PARSE_ERROR(context,
                  "Option '" << name << "' is not overlayable.\n"
                  << parsing_help<overlayable_tags_and_subgroups_list>(node));
    }

    // Check for duplicate key
    if (0 != overlaid_options.count(name)) {
      PARSE_ERROR(context,
                  "Option '" << name << "' specified twice.\n"
                  << parsing_help<overlayable_tags_and_subgroups_list>(node));
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

// template <typename OptionList, typename Group>
// template <typename T>
// void Parser<OptionList, Group>::check_lower_bound_on_size(
//     const typename T::type& t, const Context& context) const {
//   if constexpr (Options_detail::has_lower_bound_on_size<T>::value) {
//     static_assert(std::is_same_v<decltype(T::lower_bound_on_size()), size_t>,
//                   "lower_bound_on_size() is not a size_t.");
//     if (t.size() < T::lower_bound_on_size()) {
//       PARSE_ERROR(context, "Value must have at least "
//                                << T::lower_bound_on_size() << " entries, but "
//                                << t.size() << " were given.\n"
//                                << help());
//     }
//   }
// }

// template <typename OptionList, typename Group>
// template <typename T>
// void Parser<OptionList, Group>::check_upper_bound_on_size(
//     const typename T::type& t, const Context& context) const {
//   if constexpr (Options_detail::has_upper_bound_on_size<T>::value) {
//     static_assert(std::is_same_v<decltype(T::upper_bound_on_size()), size_t>,
//                   "upper_bound_on_size() is not a size_t.");
//     if (t.size() > T::upper_bound_on_size()) {
//       PARSE_ERROR(context, "Value must have at most "
//                                << T::upper_bound_on_size() << " entries, but "
//                                << t.size() << " were given.\n"
//                                << help());
//     }
//   }
// }

// template <typename OptionList, typename Group>
// template <typename T>
// inline void Parser<OptionList, Group>::check_lower_bound(
//     const typename T::type& t, const Context& context) const {
//   if constexpr (Options_detail::has_lower_bound<T>::value) {
//     static_assert(std::is_same_v<decltype(T::lower_bound()), typename T::type>,
//                   "Lower bound is not of the same type as the option.");
//     static_assert(not std::is_same_v<typename T::type, bool>,
//                   "Cannot set a lower bound for a bool.");
//     if (t < T::lower_bound()) {
//       PARSE_ERROR(context, "Value " << (MakeString{} << t)
//                                     << " is below the lower bound of "
//                                     << (MakeString{} << T::lower_bound())
//                                     << ".\n" << help());
//     }
//   }
// }

// template <typename OptionList, typename Group>
// template <typename T>
// inline void Parser<OptionList, Group>::check_upper_bound(
//     const typename T::type& t, const Context& context) const {
//   if constexpr (Options_detail::has_upper_bound<T>::value) {
//     static_assert(std::is_same_v<decltype(T::upper_bound()), typename T::type>,
//                   "Upper bound is not of the same type as the option.");
//     static_assert(not std::is_same_v<typename T::type, bool>,
//                   "Cannot set an upper bound for a bool.");
//     if (t > T::upper_bound()) {
//       PARSE_ERROR(context, "Value " << (MakeString{} << t)
//                                     << " is above the upper bound of "
//                                     << (MakeString{} << T::upper_bound())
//                                     << ".\n" << help());
//     }
//   }
// }

// template <typename OptionList, typename Group>
// template <typename TagsAndSubgroups>
// std::string Parser<OptionList, Group>::parsing_help(
//     const YAML::Node& options) const {
//   std::ostringstream os;
//   // At top level this would dump the entire input file, which is very
//   // verbose and not very informative.  At lower levels the result
//   // should be much shorter and may actually give useful context for
//   // what part of the file is being parsed.
//   if (not context_.top_level) {
//     os << "\n==== Parsing the option string:\n" << options << "\n";
//   }
//   os << help<TagsAndSubgroups>();
//   return os.str();
// }

// template <typename OptionList, typename Group>
// [[noreturn]] void Parser<OptionList, Group>::parser_error(
//     const YAML::Exception& e) const {
//   auto context = context_;
//   context.line = e.mark.line;
//   context.column = e.mark.column;
//   // Inline the top_level branch of PARSE_ERROR to avoid warning that
//   // the other branch would call terminate.  (Parser errors can only
//   // be generated at top level.)
//   ERROR(
//       "\n"
//       << context
//       << "Unable to correctly parse the input file because of a syntax error.\n"
//          "This is often due to placing a suboption on the same line as an "
//          "option, e.g.:\nDomainCreator: CreateInterval:\n  IsPeriodicIn: "
//          "[false]\n\nShould be:\nDomainCreator:\n  CreateInterval:\n    "
//          "IsPeriodicIn: [true]\n\nSee an example input file for help.");
// }

namespace Options_detail {
template <typename T, typename Metavariables, typename = std::void_t<>>
struct get_options_list {
  using type = typename T::template options<Metavariables>;
};

template <typename T, typename Metavariables>
struct get_options_list<T, Metavariables, std::void_t<typename T::options>> {
  using type = typename T::options;
};
}  // namespace Options_detail

template <typename T>
template <typename Metavariables>
T create_from_yaml<T>::create(const Option& options) {
  Parser<typename Options_detail::get_options_list<T, Metavariables>::type>
      parser(T::help);
  parser.parse(options);
  return parser.template apply_all<Metavariables>([&options](
                                                      auto parsed_options,
                                                      auto&&... args) {
    if constexpr (std::is_constructible<T, decltype(parsed_options),
                                        decltype(std::move(args))...,
                                        const Context&, Metavariables>{}) {
      return T(parsed_options, std::move(args)..., options.context(),
               Metavariables{});
    } else if constexpr (std::is_constructible<T, decltype(parsed_options),
                                               decltype(std::move(args))...,
                                               const Context&>{}) {
      return T(parsed_options, std::move(args)..., options.context());
    } else if constexpr (std::is_constructible<T, decltype(parsed_options),
                                               decltype(
                                                   std::move(args))...>{}) {
      return T(parsed_options, std::move(args)...);
    } else if constexpr (std::is_constructible<T, decltype(std::move(args))...,
                                               const Context&,
                                               Metavariables>{}) {
      return T(std::move(args)..., options.context(), Metavariables{});
    } else if constexpr (std::is_constructible<T, decltype(std::move(args))...,
                                               const Context&>{}) {
      return T(std::move(args)..., options.context());
    } else {
      return T{std::move(args)...};
    }
  });
}

// yaml-cpp doesn't handle C++11 types yet
template <typename K, typename V, typename H, typename P>
struct create_from_yaml<std::unordered_map<K, V, H, P>> {
  template <typename Metavariables>
  static std::unordered_map<K, V, H, P> create(const Option& options) {
    auto ordered = options.parse_as<std::map<K, V>, Metavariables>();
    std::unordered_map<K, V, H, P> result;
    for (auto it = ordered.begin(); it != ordered.end();) {
      auto node = ordered.extract(it++);
      result.emplace(std::move(node.key()), std::move(node.mapped()));
    }
    return result;
  }
};

namespace Options_detail {
// To get the full parse backtrace for a variant parse error the
// failure should occur inside a nested call to parse_as.  This is a
// type that will produce the correct error by failing to parse.
template <typename... T>
struct variant_parse_error {};

template <typename... T>
struct yaml_type<variant_parse_error<T...>> : yaml_type<std::variant<T...>> {};
}  // namespace Options_detail

template <typename... T>
struct create_from_yaml<Options::Options_detail::variant_parse_error<T...>> {
  template <typename Metavariables>
  [[noreturn]] static Options::Options_detail::variant_parse_error<T...> create(
      const Option& options) {
    throw YAML::BadConversion(options.node().Mark());
  }
};

template <typename... T>
struct create_from_yaml<std::variant<T...>> {
  using Result = std::variant<T...>;
  static_assert(std::is_same_v<tmpl::list<T...>,
                               tmpl::remove_duplicates<tmpl::list<T...>>>,
                "Cannot parse variants with duplicate types.");

  template <typename Metavariables>
  static Result create(const Option& options) {
    Result result{};
    bool constructed = false;
    const auto try_parse =
        [&constructed, &options, &result](auto alternative_v) {
      using Alternative = tmpl::type_from<decltype(alternative_v)>;
      if (constructed) {
        return;
      }
      try {
        result = options.parse_as<Alternative, Metavariables>();
        constructed = true;
      } catch (...) {
        // This alternative failed, but a later one may succeed.
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(try_parse(tmpl::type_<T>{}));
    if (not constructed) {
      options
          .parse_as<Options_detail::variant_parse_error<T...>, Metavariables>();
    }
    return result;
  }
};

template <typename Metavariables, typename TagList, typename Group>
void create_all_options(const Parser<TagList, Group>& options) {
  options.template apply<TagList, Metavariables>([](auto... args) {
        (void)std::initializer_list<char>{((void)args, '0')...};
      });
}

template <typename TagList, typename Metavariables>
tuples::tagged_tuple_from_typelist<TagList> create_all_options() {
  Parser<tmpl::remove<TagList, Tags::InputSource>>
        options(Metavariables::help);

  create_all_options<Metavariables>(options);
  options.help();
  options.parse_file("");
  options.parse("");

  tuples::tagged_tuple_from_typelist<TagList> result =
      options.template apply<TagList, Metavariables>([](auto... args) {
          return tuples::tagged_tuple_from_typelist<TagList>(
              std::move(args)...);
        });
  return result;
}
}  // namespace Options

/// \cond
template <typename T, typename Metavariables>
struct YAML::convert<Options::Options_detail::CreateWrapper<T, Metavariables>> {
  static bool decode(
      const Node& node,
      Options::Options_detail::CreateWrapper<T, Metavariables>& rhs) {
    Options::Context context;
    context.top_level = false;
    context.append("While creating a " + pretty_type::name<T>());
    Options::Option options(node, std::move(context));
    rhs = Options::Options_detail::CreateWrapper<T, Metavariables>{
        Options::create_from_yaml<T>::template create<Metavariables>(options)};
    return true;
  }
};
/// \endcond

#include "Options/Factory.hpp"
