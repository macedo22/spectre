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
 * \brief Computes the spatial derivative of the spatial christoffel symbols of
 * the second kind from the inverse spatial metric, the spatial derivative of
 * the inverse spatial metric, the first spatial derivative of the spatial
 * metric, and the second spatial derivative of the spatial metric.
 *
 * \details Computes the derivative as:
 * \f{align}
 *     \partial_k \Gamma^m{}_{ij} &=
 *       -\partial_k \gamma^{ij} (\partial_i \gamma_{jl} +
 *       \partial_j \gamma_{il} - \partial_l \gamma_{ij}) +
 *       \gamma^{ml}(\partial_{(k} \partial_{i)jl} +
 *       \partial_{(k} \partial_{j)il} - \partial_{(k} \partial_{l)ij})
 * \f}
 * where \f$\gamma^{ij}\f$, \f$\partial_k \gamma^{ij}\f$,
 * \f$\partial_k \gamma_{ij}\f$, and \f$\partial_{(l} \partial_{k)ij}\f$ are the
 * inverse spatial metric, the spatial derivative of the inverse spatial metric,
 * the first spatial derivative of the spatial metric, and the second derivative
 * of the spatial metric, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
void deriv_spatial_christoffel_second_kind(
    const gsl::not_null<tnsr::iJkk<DataType, Dim, Frame>*> d_christoffel,
    const tnsr::iJJ<DataType, Dim, Frame>& d_inverse_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& d_spatial_metric,
    const Tensor<DataType, Symmetry<2, 2, 1, 1>,
                 index_list<SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>>>&
        d2_spatial_metric) noexcept;

template <size_t Dim, typename Frame, typename DataType>
tnsr::iJkk<DataType, Dim, Frame> deriv_spatial_christoffel_second_kind(
    const tnsr::iJJ<DataType, Dim, Frame>& d_inverse_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& d_spatial_metric,
    const Tensor<DataType, Symmetry<2, 2, 1, 1>,
                 index_list<SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>>>&
        d2_spatial_metric) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item to get the spatial derivative of the spatial christoffel
 * symbols of the second kind from the inverse spatial metric, the spatial
 * derivative of the inverse spatial metric, the first spatial derivative of the
 * spatial metric, and the second spatial derivative of the spatial metric.
 *
 * \details See `deriv_spatial_christoffel_second_kind()`. Can be retrieved
 * using `gr::Tags::DerivSpatialChristoffelSecondKind`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct DerivSpatialChristoffelSecondKindCompute
    : gr::Tags::DerivSpatialChristoffelSecondKind<Dim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::deriv<gr::Tags::InverseSpatialMetric<Dim, Frame, DataType>,
                    tmpl::size_t<Dim>, Frame>,
      gr::Tags::InverseSpatialMetric<Dim, Frame, DataType>,
      ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame, DataType>,
                    tmpl::size_t<Dim>, Frame>,
      ::Tags::deriv<::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame, DataType>,
                                  tmpl::size_t<Dim>, Frame>,
                    tmpl::size_t<Dim>, Frame>>;

  using return_type = tnsr::iJkk<DataType, Dim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::iJkk<DataType, Dim, Frame>*>,
      const tnsr::iJJ<DataType, Dim, Frame>&,
      const tnsr::II<DataType, Dim, Frame>&,
      const tnsr::ijj<DataType, Dim, Frame>&,
      const Tensor<DataType, Symmetry<2, 2, 1, 1>,
                   index_list<SpatialIndex<Dim, UpLo::Lo, Frame>,
                              SpatialIndex<Dim, UpLo::Lo, Frame>,
                              SpatialIndex<Dim, UpLo::Lo, Frame>,
                              SpatialIndex<Dim, UpLo::Lo, Frame>>>&) noexcept>(
      &gr::deriv_spatial_christoffel_second_kind<Dim, Frame, DataType>);

  using base =
      gr::Tags::DerivSpatialChristoffelSecondKind<Dim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace gr
