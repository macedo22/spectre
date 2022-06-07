// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

/// \ingroup TypeTraitsGroup
/// C++ STL code present in C++20
namespace cpp20 {
/// \ingroup TypeTraitsGroup
/// \brief Determine whether or not the function call occurs within a
/// constant-evaluated context
///
/// \details
/// Useful for tailoring code for compile time or runtime depending on when the
/// code is evaluated
constexpr bool is_constant_evaluated() {
  return __builtin_is_constant_evaluated();
}

/// \ingroup TypeTraitsGroup
template <class T>
struct remove_cvref {
  // clang-tidy use using instead of typedef
  typedef std::remove_cv_t<std::remove_reference_t<T>> type;  // NOLINT
};

/// \ingroup TypeTraitsGroup
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;
}  // namespace cpp20
