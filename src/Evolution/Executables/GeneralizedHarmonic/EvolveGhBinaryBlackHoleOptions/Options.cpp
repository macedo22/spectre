// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/GeneralizedHarmonic/EvolveGhBinaryBlackHole.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/ParseOptions.tpp"
#include "Parallel/MainOptionList.hpp"
#include "Utilities/TaggedTuple.hpp"

template
tuples::tagged_tuple_from_typelist<typename Parallel::get_main_option_list<EvolutionMetavars>::type> Options::create_all_options<
    typename Parallel::get_main_option_list<EvolutionMetavars>::type,
    EvolutionMetavars>();

// namespace {
// using Metavariables = EvolutionMetavars;
// using option_list = typename Parallel::get_main_option_list<Metavariables>::type;

// tmpl::for_each<next_option>([this, &alternative_number, &choice, &choices,
//                                    &func, &result](auto alternative) {
//         using Alternative = tmpl::type_from<decltype(alternative)>;
//         if (choice == alternative_number++) {
//           result = this->call_with_chosen_alternatives_impl<
//               ChosenOptions, tmpl::append<Alternative, remaining_options>>(
//               std::forward<F>(func), std::move(choices));
//         }
//       });
// tmpl::for_each<next_option>([this, &alternative_number, &choice, &choices,
//                                    &func, &result](auto alternative) {
//         using Alternative = tmpl::type_from<decltype(alternative)>;
//         if (choice == alternative_number++) {
//           result = this->call_with_chosen_alternatives_impl<
//               ChosenOptions, tmpl::append<Alternative, remaining_options>>(
//               std::forward<F>(func), std::move(choices));
//         }
//       });

// template <typename TagList>
// struct instantiate_apply_impl;

// template <typename... Tags>
// struct instantiate_apply_impl<tmpl::list<Tags...>> {
//   template <typename Metavariables>
//   static decltype(auto) apply(const Options& opts, F&& func) {
//     return func(opts.template get<Tags, Metavariables>()...);
//   }
// };


// template <typename TagList, typename Metavariables>
// void instantiate_apply() {
//   Options::Parser<tmpl::remove<TagList, Options::Tags::InputSource>>
//         options(Metavariables::help);

//   options.template apply<TagList, Metavariables>([](auto... args) {
//         (void)std::initializer_list<char>{((void)args, '0')...};
//       });
// }
// }  // namespace

// tmpl::for_each<next_option>([this, &alternative_number, &choice, &choices,
//                                    &func, &result](auto alternative) {
//         using Alternative = tmpl::type_from<decltype(alternative)>;
//         if (choice == alternative_number++) {
//           result = this->call_with_chosen_alternatives_impl<
//               ChosenOptions, tmpl::append<Alternative, remaining_options>>(
//               std::forward<F>(func), std::move(choices));
//         }
//       }); 

// template <typename OptionList, typename Group>
// template <typename Tag, typename Metavariables>
// typename Tag::type Parser<OptionList, Group>::get() const;

// template <typename... Tags>
// struct apply_helper<tmpl::list<Tags...>> {
//   template <typename Metavariables, typename Options, typename F>
//   static decltype(auto) apply(const Options& opts, F&& func) {
//     return func(opts.template get<Tags, Metavariables>()...);
//   }
// };

