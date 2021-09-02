
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

// TODO: compute the extrinsic curvature? not have these tags?

namespace CCZ4 {
template <size_t SpatialDim, typename Frame, typename DataType>
void extrinsic_curvature(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>& normal_one_form,
    const tnsr::ii<DataType, SpatialDim, Frame>& grad_unit_normal_one_form,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        christoffel_2nd_kind) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const tnsr::i<DataType, SpatialDim, Frame>& normal_one_form,
    const tnsr::ii<DataType, SpatialDim, Frame>& grad_unit_normal_one_form,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        christoffel_2nd_kind) noexcept;

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the trace of the extrinsic curvature \f$\K\f$ used by the
 * CCZ4 formulation of Einstein's equations.
 *
 * \details If \f$ \gamma^{ij}\f$ is the inverse spatial metric and
 * \f$\K_{ij}\f$ is the extrinsic curvature, the trace is computed as
 *
 * \f{align}
 *     K &= K_{ij} \gamma^{ij}
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void trace_extrinsic_curvature(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> trace_extrinsic_curvature(
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item for the extrinsic curvature \f$\K_{ij}\f$ used by the
 * CCZ4 formulation of Einstein's equations.
 *
 * \details See `extrinsic_curvature()`. Can be retrieved using
 * `CCZ4::Tags::ExtrinsicCurvature`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct ExtrinsicCurvatureCompute
    : ExtrinsicCurvature<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags = tmpl::list<>;

  using return_type = tnsr::ii<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>) noexcept>(
      &extrinsic_curvature<SpatialDim, Frame, DataType>);

  using base = ExtrinsicCurvature<SpatialDim, Frame, DataType>;
};

/*!
 * \brief Compute item for the trace of the extrinsic curvature \f$\K\f$ used by
 * the CCZ4 formulation of Einstein's equations.
 *
 * \details See `trace_extrinsic_curvature()`. Can be retrieved using
 * `CCZ4::Tags::TraceExtrinsicCurvature`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct TraceExtrinsicCurvatureCompute : TraceExtrinsicCurvature<DataType>,
                                        db::ComputeTag {
  using argument_tags =
      tmpl::list<ExtrinsicCurvature<SpatialDim, Frame, DataType>,
                 ::gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function =
      static_cast<void (*)(const gsl::not_null<Scalar<DataType>*>) noexcept>(
          &trace_extrinsic_curvature<SpatialDim, Frame, DataType>);

  using base = TraceExtrinsicCurvature<DataType>;
};
}  // namespace Tags
}  // namespace CCZ4
