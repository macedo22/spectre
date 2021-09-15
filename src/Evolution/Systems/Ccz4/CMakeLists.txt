# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Ccz4)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DerivLapse.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DerivLapse.hpp
  Tags.hpp
  TagsDeclarations.hpp
  )
