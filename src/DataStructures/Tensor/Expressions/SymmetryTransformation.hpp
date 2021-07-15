// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

// TODO: update description
/// \file
/// Defines functions and metafunctions used for helping evaluate
/// TensorExpression equations where generic spatial indices are used for
/// spacetime indices

namespace TensorExpressions {
namespace detail {
// @{
// TODO: document
template <
    size_t NumIndices, size_t NumSpatialSpacetimeIndices,
    size_t NumConcreteTimeIndices,
    Requires<(NumIndices >= 2 and (NumSpatialSpacetimeIndices != 0 or
                                   NumConcreteTimeIndices != 0))> = nullptr>
constexpr std::array<std::int32_t, NumIndices>
get_transformed_spacetime_symmetry(
    const std::array<std::int32_t, NumIndices>& symmetry,
    const std::array<size_t, NumSpatialSpacetimeIndices>&
        spatial_spacetime_index_positions,
    const std::array<size_t, NumConcreteTimeIndices>&
        concrete_time_index_positions) noexcept {
  std::array<std::int32_t, NumIndices> transformed_spacetime_symmetry{};
  const std::int32_t max_symm_value =
      static_cast<std::int32_t>(*alg::max_element(symmetry));
  for (size_t i = 0; i < NumIndices; i++) {
    gsl::at(transformed_spacetime_symmetry, i) = gsl::at(symmetry, i);
  }
  for (size_t i = 0; i < NumSpatialSpacetimeIndices; i++) {
    gsl::at(transformed_spacetime_symmetry,
            gsl::at(spatial_spacetime_index_positions, i)) += max_symm_value;
  }
  for (size_t i = 0; i < NumConcreteTimeIndices; i++) {
    gsl::at(transformed_spacetime_symmetry,
            gsl::at(concrete_time_index_positions, i)) += 2 * max_symm_value;
  }

  return transformed_spacetime_symmetry;
}

template <size_t NumIndices, size_t NumSpatialSpacetimeIndices,
          size_t NumConcreteTimeIndices,
          Requires<(NumIndices < 2 or (NumSpatialSpacetimeIndices == 0 and
                                       NumConcreteTimeIndices == 0))> = nullptr>
constexpr std::array<std::int32_t, NumIndices>
get_transformed_spacetime_symmetry(
    const std::array<std::int32_t, NumIndices>& symmetry,
    const std::array<size_t, NumSpatialSpacetimeIndices>&
    /*spatial_spacetime_index_positions*/,
    const std::array<size_t, NumConcreteTimeIndices>&
    /*concrete_time_index_positions*/) noexcept {
  return symmetry;
}
// @}
}  // namespace detail
}  // namespace TensorExpressions
