// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macro ASSERT.

#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include "Utilities/ErrorHandling/AbortWithErrorMessage.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Assert that an expression should be true.
 *
 * If the preprocessor macro SPECTRE_DEBUG is defined and the expression is
 * false, an error message is printed to the standard error stream, and the
 * program aborts. ASSERT should be used to catch coding errors as it does
 * nothing in production code.
 * \param a the expression that must be true
 * \param m the error message as an ostream
 */
#define ASSERT(a, m)                                                           \
  do {                                                                         \
    if (false) {                                                               \
      static_cast<void>(a);                                                    \
      const ScopedFpeState disable_fpes_ASSERT(false);                         \
      std::ostringstream avoid_name_collisions_ASSERT;                         \
      /* clang-tidy: macro arg in parentheses */                               \
      avoid_name_collisions_ASSERT << std::setprecision(18) << std::scientific \
                                   << m; /* NOLINT */                          \
      static_cast<void>(avoid_name_collisions_ASSERT);                         \
    }                                                                          \
  } while (false)
