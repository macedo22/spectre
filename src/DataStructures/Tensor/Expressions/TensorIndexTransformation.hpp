// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions useful for converting tensor multi-indices according to a
/// different generic index order

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <limits>

#include "DataStructures/Tensor/Expressions/ConcreteTimeIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace TensorExpressions {
namespace TensorIndexTransformation_detail {
static constexpr size_t time_index_position_placeholder =
    std::numeric_limits<size_t>::max();
}  // namespace TensorIndexTransformation_detail

// TODO: review and update description, if needed, since this now can handle
// differently-size array arguments
/// \brief Computes a transformation from one generic tensor index order to
/// another
///
/// \details
/// The elements of the transformation are the positions of the second list of
/// generic indices in the first list of generic indices. Put another way, for
/// some `i`, `tensorindices2[i] == tensorindices1[index_transformation[i]]`.
///
/// Here is an example of what the algorithm does:
///
/// Transformation between (1) \f$R_{cab}\f$ and (2) \f$S_{abc}\f$
/// `tensorindices1`:
/// \code
/// {2, 0, 1} // TensorIndex values for {c, a, b}
/// \endcode
/// `tensorindices2`:
/// \code
/// {0, 1, 2} // TensorIndex values for {a, b, c}
/// \endcode
/// returned `tensorindex_transformation`:
/// \code
/// {1, 2, 0} // positions of S' indices {a, b, c} in R's indices {c, a, b}
/// \endcode
///
/// \tparam NumIndices1 the number of indices for the first generic index order
/// \tparam NumIndices2 the number of indices for the second generic index order
/// \param tensorindices1 the TensorIndex values of the first generic index
/// order
/// \param tensorindices2 the TensorIndex values of the second generic index
/// order
/// \return a transformation from the first generic index order to the second
template <size_t NumIndices1, size_t NumIndices2>
SPECTRE_ALWAYS_INLINE constexpr std::array<size_t, NumIndices2>
compute_tensorindex_transformation(
    const std::array<size_t, NumIndices1>& tensorindices1,
    const std::array<size_t, NumIndices2>& tensorindices2) noexcept {
  std::array<size_t, NumIndices2> tensorindex_transformation{};
  for (size_t i = 0; i < NumIndices2; i++) {
    gsl::at(tensorindex_transformation, i) =
        detail::is_concrete_time_index_value(gsl::at(tensorindices2, i))
            ? TensorIndexTransformation_detail::time_index_position_placeholder
            : static_cast<size_t>(std::distance(
                  tensorindices1.begin(),
                  alg::find(tensorindices1, gsl::at(tensorindices2, i))));
  }
  return tensorindex_transformation;
}

// TODO: review and update description, if needed, since this now can handle
// differently-size array arguments
/// \brief Computes the tensor multi-index that is equivalent to a given tensor
/// multi-index, according to the differences in their generic index orders
///
/// \details
/// Here is an example of what the algorithm does:
///
/// Transform (input) multi-index of \f$R_{cab}\f$ to the equivalent (output)
/// multi-index of \f$S_{abc}\f$
/// `tensorindex_transformation`:
/// \code
/// {1, 2, 0} // positions of S' indices {a, b, c} in R's indices {c, a, b}
/// \endcode
/// `input_multi_index`:
/// \code
/// {3, 4, 5} // i.e. c = 3, a = 4, b = 5
/// \endcode
/// returned equivalent `output_multi_index`:
/// \code
/// {4, 5, 3} // i.e. a = 4, b = 5, c = 3
/// \endcode
///
/// \tparam NumIndicesIn the number of indices
/// \tparam NumIndicesOut the number of indices
/// \param input_multi_index the input tensor multi-index to transform
/// \param tensorindex_transformation the positions of the output's generic
/// indices in the input's generic indices (see example in details)
/// \return the output tensor multi-index that is equivalent to
/// `input_multi_index`, according to generic index order differences
// (`tensorindex_transformation`)
template <size_t NumIndicesIn, size_t NumIndicesOut>
SPECTRE_ALWAYS_INLINE constexpr std::array<size_t, NumIndicesOut>
transform_multi_index(const std::array<size_t, NumIndicesIn>& input_multi_index,
                      const std::array<size_t, NumIndicesOut>&
                          tensorindex_transformation) noexcept {
  std::array<size_t, NumIndicesOut> output_multi_index =
      make_array<NumIndicesOut, size_t>(0);
  for (size_t i = 0; i < NumIndicesOut; i++) {
    gsl::at(output_multi_index, i) =
        (gsl::at(tensorindex_transformation, i) ==
         TensorIndexTransformation_detail::time_index_position_placeholder)
            ? 0
            : gsl::at(input_multi_index,
                      gsl::at(tensorindex_transformation, i));
    gsl::at(input_multi_index, gsl::at(tensorindex_transformation, i));
  }
  return output_multi_index;
}
}  // namespace TensorExpressions
