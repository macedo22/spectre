// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the spatial derivative of inverse spatial metric from the
 * inverse spatial metric and the spatial derivative of the spatial metric.
 *
 * \details Computes the derivative as:
 * \f{align}
 *     \partial_k \gamma^{ij} &= -\gamma^{in} \gamma^{mj}
 *                 \partial_k \gamma_{nm}
 * \f}
 * where \f$\gamma^{ij}\f$ and \f$\partial_k \gamma_{ij}\f$ are the inverse
 * spatial metric and spatial derivative of the spatial metric, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
void deriv_inverse_spatial_metric(
    const gsl::not_null<tnsr::iJJ<DataType, Dim, Frame>*>
        d_inverse_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& d_spatial_metric) noexcept;

template <size_t Dim, typename Frame, typename DataType>
tnsr::iJJ<DataType, Dim, Frame> deriv_inverse_spatial_metric(
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& d_spatial_metric) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item to get the spatial derivative of inverse spatial metric
 * from the inverse spatial metric and the spatial derivative of the spatial
 * metric.
 *
 * \details See `deriv_inverse_spatial_metric()`. Can be retrieved using
 * `gr::Tags::DerivInverseSpatialMetric`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct DerivInverseSpatialMetricCompute
    : gr::Tags::DerivInverseSpatialMetric<Dim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<Dim, Frame, DataType>,
                 ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame, DataType>,
                               tmpl::size_t<Dim>, Frame>>;

  using return_type = tnsr::iJJ<DataType, Dim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::iJJ<DataType, Dim, Frame>*>,
      const tnsr::II<DataType, Dim, Frame>&,
      const tnsr::ijj<DataType, Dim, Frame>&) noexcept>(
      &gr::deriv_inverse_spatial_metric<Dim, Frame, DataType>);

  using base = gr::Tags::DerivInverseSpatialMetric<Dim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace gr
