
// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CCZ4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace CCZ4 {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the trace-free part of the extrinsic curvature
 * \f$\widetilde{A}_{ij}\f$ used by the CCZ4 formulation of Einstein's
 * equations.
 *
 * \details If \f$\phi^2\f$ is the square of the conformal factor, \f$K_{ij}\f$
 * is the extrinsic curvature, \f$K\f$ is the trace of the extrinsic curvature,
 * and \f$\\gamma_{ij}\f$ is the spatial metric, then \f$\widetilde{A}_{ij}\f$
 * is computed as
 *
 * \f{align}
 *     \widetilde{A}_{ij} &= \phi^2 (K_{ij} - \frac{1}{3} K \gamma_{ij})
 * \f}
 */
template <size_t Dim, typename Frame, typename DataType>
void a_tilde(const gsl::not_null<tnsr::ij<DataType, Dim, Frame>*> a_tilde,
             const Scalar<DataType>& phi_squared,
             const tnsr::ii<DataType, Dim, Frame>& extrinsic_curvature,
             const Scalar<DataType>& trace_extrinsic_curvature,
             const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept;

template <size_t Dim, typename Frame, typename DataType>
tnsr::ij<DataType, Dim, Frame> a_tilde(
    const Scalar<DataType>& phi_squared,
    const tnsr::ii<DataType, Dim, Frame>& extrinsic_curvature,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item for the trace-free part of the extrinsic curvature
 * \f$\widetilde{A}_{ij}\f$ used by the CCZ4 formulation of Einstein's
 * equations.
 *
 * \details See `a_tilde()`. Can be retrieved using `CCZ4::Tags::ATilde`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct ATildeCompute : ATilde<SpatialDim, Frame, DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<PhiSquared<DataType>,
                 ExtrinsicCurvature<SpatialDim, Frame, DataType>,
                 TraceExtrinsicCurvature<DataType>,
                 ::gr::Tags::SpatialMetric<SpatialDim, Frame, DataType>>;

  using return_type = tnsr::ij<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::ij<DataType, SpatialDim, Frame>*>,
      const Scalar<DataType>&, const tnsr::ii<DataType, SpatialDim, Frame>&,
      const Scalar<DataType>&,
      const tnsr::ii<DataType, SpatialDim, Frame>&) noexcept>(
      &a_tilde<SpatialDim, Frame, DataType>);

  using base = ATilde<SpatialDim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace CCZ4
