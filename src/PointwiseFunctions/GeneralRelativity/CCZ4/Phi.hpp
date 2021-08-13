
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
 * \brief Computes the conformal factor \f$\phi\f$ used by the CCZ4 formulation
 * of Einstein's equations.
 *
 * \details If \f$ \gamma_{ij}\f$ is the spatial metric, then \f$\phi\f$ is
 * computed as
 *
 * \f{align}
 *     \phi &= (det(\gamma_{ij}))^{-1/3}
 * \f}
 */
template <typename DataType>
void phi(const gsl::not_null<Scalar<DataType>*> phi,
         const Scalar<DataType>& det_spatial_metric) noexcept;

template <typename DataType>
Scalar<DataType> phi(const Scalar<DataType>& det_spatial_metric) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item for the conformal factor \f$\phi\f$ used by the CCZ4
 * formulation of Einstein's equations.
 *
 * \details See `phi()`. Can be retrieved using `CCZ4::Tags::Phi`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct PhiCompute : Phi<DataType>, db::ComputeTag {
  using argument_tags = tmpl::list<::gr::Tags::DetSpatialMetric<DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const Scalar<DataType>&) noexcept>(
      &phi<SpatialDim, Frame, DataType>);

  using base = Phi<DataType>;
};
}  // namespace Tags
}  // namespace CCZ4
