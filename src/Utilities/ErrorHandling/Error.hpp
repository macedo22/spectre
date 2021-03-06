// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macro ERROR.

#pragma once

#include <sstream>
#include <string>

#include "Utilities/ErrorHandling/AbortWithErrorMessage.hpp"
#include "Utilities/ErrorHandling/Breakpoint.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/System/Abort.hpp"

/*!
 * \ingroup ErrorHandlingGroup
 * \brief prints an error message to the standard error stream and aborts the
 * program.
 *
 * ERROR should not be used for coding errors, but instead for user errors
 * or failure modes of numerical algorithms. An acceptable use for error is also
 * in the default case of a switch statement.
 * \param m an arbitrary output stream.
 */
// isocpp.org recommends using an `if (true)` instead of a `do
// while(false)` for macros because the latter can mess with inlining
// in some (old?) compilers:
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-multi-stmts
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-if
// However, Intel's reachability analyzer (as of version 16.0.3
// 20160415) can't figure out that the else branch and everything
// after it is unreachable, causing warnings (and possibly suboptimal
// code generation).
#define ERROR(m)                                                            \
  do {                                                                      \
    disable_floating_point_exceptions();                                    \
    std::ostringstream avoid_name_collisions_ERROR;                         \
    /* clang-tidy: macro arg in parentheses */                              \
    avoid_name_collisions_ERROR << m; /* NOLINT */                          \
    abort_with_error_message(__FILE__, __LINE__,                            \
                             static_cast<const char*>(__PRETTY_FUNCTION__), \
                             avoid_name_collisions_ERROR.str());            \
  } while (false)

/*!
 * \ingroup ErrorHandlingGroup
 * \brief prints an error message to the standard error and aborts the
 * program.
 *
 * CERROR is just like ERROR and so the same guidelines apply. However, because
 * it does not use std::stringstream it can be used in some constexpr
 * functions where ERROR cannot be.
 * \param m error message as a string, may need to use string literals
 */
#define CERROR(m)                                                             \
  do {                                                                        \
    breakpoint();                                                             \
    sys::abort("\n################ ERROR ################\nLine: "s +         \
               std::to_string(__LINE__) + " of file '"s + __FILE__ + "'\n"s + \
               m + /* NOLINT */                                               \
               "\n#######################################\n"s);               \
  } while (false)

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Same as ERROR but does not print a backtrace. Intended to be used for
 * user errors, such as incorrect values in an input file.
 */
#define ERROR_NO_TRACE(m)                                                  \
  do {                                                                     \
    disable_floating_point_exceptions();                                   \
    std::ostringstream avoid_name_collisions_ERROR;                        \
    /* clang-tidy: macro arg in parentheses */                             \
    avoid_name_collisions_ERROR << m; /* NOLINT */                         \
    abort_with_error_message_no_trace(                                     \
        __FILE__, __LINE__, static_cast<const char*>(__PRETTY_FUNCTION__), \
        avoid_name_collisions_ERROR.str());                                \
  } while (false)
