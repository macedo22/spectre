// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <type_traits>

#include "DataStructures/DataBox/TagTraits.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/*!
 * \ingroup DataBoxTagsGroup
 * \brief Tag used to retrieve the DataBox from the `db::get` function
 *
 * The main use of this tag is to allow fetching the DataBox from itself. The
 * primary use case is to allow an invokable to take a DataBox as an argument
 * when called through `db::apply`.
 *
 * \snippet Test_DataBox.cpp databox_self_tag_example
 */
struct DataBox {
  // Trick to get friend function declaration to compile but a const
  // NoSuchtype****& is rather useless
  using type = NoSuchType****;
};
}  // namespace Tags

namespace db {

namespace detail {
template <typename Tag, typename TagList>
struct first_matching_tag_impl {
  // type list from first match to the end of TagList
  using find_result =
      typename tmpl::find<TagList, std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>;

  static_assert(tmpl::size<find_result>::value != 0,
                "Could not find the DataBox tag in the list of DataBox tags. "
                "The first template parameter of 'first_matching_tag_impl' is "
                "the tag that cannot be found and the second is the list of "
                "tags being searched.");

  using type = typename tmpl::front<find_result>;
};

template <typename TagList>
struct first_matching_tag_impl<::Tags::DataBox, TagList> {
  using type = ::Tags::DataBox;
};

template <typename TagList, typename Tag>
using first_matching_tag = typename first_matching_tag_impl<Tag, TagList>::type;

template <typename Tag, typename TagList>
struct number_of_matching_tags_impl {
  // find first match, then search for matches from there to the end of TagList
  static constexpr size_t value =
      tmpl::count_if<TagList, std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>::value;
};

template <typename TagList>
struct number_of_matching_tags_impl<::Tags::DataBox, TagList> {
  static constexpr size_t value = 1;
};

template <typename TagList, typename Tag>
constexpr auto number_of_matching_tags =
    number_of_matching_tags_impl<Tag, TagList>::value;

template <typename TagList, typename Tag>
struct has_unique_matching_tag
    : std::integral_constant<bool, number_of_matching_tags<TagList, Tag> == 1> {
};

template <typename TagList, typename Tag>
using has_unique_matching_tag_t =
    typename has_unique_matching_tag<TagList, Tag>::type;

template <typename TagList, typename Tag>
constexpr bool has_unique_matching_tag_v =
    has_unique_matching_tag<TagList, Tag>::value;

template <typename TagList, typename Tag>
struct has_no_matching_tag
    : std::integral_constant<bool, number_of_matching_tags<TagList, Tag> == 0> {
};

template <typename TagList, typename Tag>
using has_no_matching_tag_t = typename has_no_matching_tag<TagList, Tag>::type;

template <typename TagList, typename Tag>
constexpr bool has_no_matching_tag_v = has_no_matching_tag<TagList, Tag>::value;

template <typename T>
struct ConvertToConst {
  using type = const T&;
};

template <typename T>
struct ConvertToConst<std::unique_ptr<T>> {
  using type = const T&;
};

template <typename Tag, typename TagsList, bool = db::is_base_tag_v<Tag>>
struct const_item_type_impl {
  using type = typename db::detail::ConvertToConst<
      std::decay_t<typename Tag::type>>::type;
};

template <typename Tag, typename TagsList>
struct const_item_type_impl<Tag, TagsList, true> {
  using type = typename db::detail::ConvertToConst<std::decay_t<
      typename db::detail::first_matching_tag<TagsList, Tag>::type>>::type;
};
}  // namespace detail

template <typename Tag, typename TagsList>
using const_item_type =
    typename detail::const_item_type_impl<Tag, TagsList>::type;
}  // namespace db
