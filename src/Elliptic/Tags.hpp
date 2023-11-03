// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/Serialize.hpp"

/// Functionality related to solving elliptic partial differential equations
namespace elliptic {

namespace OptionTags {

template <typename BackgroundType>
struct Background {
  static Options::String help();
  using type = std::unique_ptr<BackgroundType>;
};

template <typename InitialGuessType>
struct InitialGuess {
  static Options::String help();
  using type = std::unique_ptr<InitialGuessType>;
};

}  // namespace OptionTags

namespace Tags {

/// The variable-independent part of the elliptic equations, e.g. the
/// fixed-sources \f$f(x)\f$ in a Poisson equation \f$-\Delta u=f(x)\f$, the
/// matter-density in a TOV-solve or the conformal metric in an XCTS solve.
template <typename BackgroundType>
struct Background : db::SimpleTag {
  using type = std::unique_ptr<BackgroundType>;
  using option_tags = tmpl::list<OptionTags::Background<BackgroundType>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) {
    return deserialize<type>(serialize<type>(value).data());
  }
};

/// The initial guess for the elliptic solve.
template <typename InitialGuessType>
struct InitialGuess : db::SimpleTag {
  using type = std::unique_ptr<InitialGuessType>;
  using option_tags = tmpl::list<OptionTags::InitialGuess<InitialGuessType>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) {
    return deserialize<type>(serialize<type>(value).data());
  }
};

}  // namespace Tags
}  // namespace elliptic
