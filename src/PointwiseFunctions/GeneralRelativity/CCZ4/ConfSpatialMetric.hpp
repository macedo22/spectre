
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

/// \cond
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace CCZ4 {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the conformal spatial metric \f$\widetilde{\gamma}_{ij}\f$
 * used by the CCZ4 formulation of Einstein's equations.
 *
 * \details If \f$\phi\f$ is the conformal factor and \f$\gamma_{ij}\f$ is the
 * spatial metric, then \f$\\widetilde{\gamma}_{ij}\f$ is computed as
 *
 * \f{align}
 *     \widetilde{\gamma}_{ij} &= \f$\phi^2 \gamma_{ij}\f$
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void conformal_spatial_metric(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        conf_spatial_metric,
    const Scalar<DataType>& phi_squared,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> conformal_spatial_metric(
    const Scalar<DataType>& phi_squared,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item for the conformal spatial metric
 * \f$\widetilde{\gamma}_{ij}\f$ used by the CCZ4 formulation of Einstein's
 * equations.
 *
 * \details See `conformal_spatial_metric()`. Can be retrieved using
 * `CCZ4::Tags::ConfSpatialMetric`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct ConfSpatialMetricCompute
    : ConfSpatialMetric<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<PhiSquared<DataType>,
                 ::gr::Tags::SpatialMetric<SpatialDim, Frame, DataType>>;

  using return_type = tnsr::ii<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>,
      const tnsr::ii<DataVector, SpatialDim, Frame>&) noexcept>(
      &conformal_spatial_metric<SpatialDim, Frame, DataType>);

  using base = ConfSpatialMetricCompute<SpatialDim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace CCZ4
