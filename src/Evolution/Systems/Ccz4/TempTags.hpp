// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"

namespace Ccz4 {
namespace Tags {
/*!
 * \brief The CCZ4 temporary expression
 * \f$\hat{\Gamma}^i - \tilde{\Gamma}^i\f$
 *
 * \details Here, \f$\hat{\Gamma}^{i}\f$ is the CCZ4 identity defined by
 * `Ccz4::Tags::GammaHat` and \f$\tilde{\Gamma}^{i}\f$ is the contraction of the
 * conformal spatial Christoffel symbols of the second kind defined by
 * `Ccz4::Tags::ContractedConformalChristoffelSecondKind`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct GammaHatMinusContractedConformalChristoffel : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};
}  // namespace Tags
}  // namespace Ccz4
