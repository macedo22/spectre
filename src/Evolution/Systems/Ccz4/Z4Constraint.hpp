// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
 * \brief Computes the spatial part of the Z4 constraint
 *
 * \details Computes the constraint as:
 *
 * \f{align}
 *     Z_i &= \frac{1}{2} \tilde{\gamma}_{ij} \left(
 *         \hat{\Gamma}^j - \tilde{\Gamma}^j\right)
 * \f}
 *
 * where \f$\tilde{\gamma}_{ij}\f$ is the conformal spatial metric defined by
 * `Ccz4::Tags::ConformalMetric`, \f$\tilde{\Gamma}^i\f$ is the contraction of
 * the conformal spatial christoffel symbols of the second kind defined by
 * `Ccz4::Tags::ContractedConformalChristoffelSecondKind`, and
 * \f$\hat{\Gamma}^i\f$ is the CCZ4 identity defined by `Ccz4::Tags::GammaHat`.
 */
template <size_t Dim, typename Frame, typename DataType>
void spatial_z4_constraint(
    const gsl::not_null<tnsr::i<DataType, Dim, Frame>*> result,
    const gsl::not_null<tnsr::I<DataType, Dim, Frame>*> buffer,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::I<DataType, Dim, Frame>&
        contracted_conformal_christoffel_second_kind,
    const tnsr::I<DataType, Dim, Frame>& gamma_hat);

template <size_t Dim, typename Frame, typename DataType>
tnsr::i<DataType, Dim, Frame> spatial_z4_constraint(
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::I<DataType, Dim, Frame>&
        contracted_conformal_christoffel_second_kind,
    const tnsr::I<DataType, Dim, Frame>& gamma_hat);
/// @}
}  // namespace Ccz4
