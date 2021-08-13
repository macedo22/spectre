
// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CCZ4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace CCZ4 {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the square of the conformal factor, \f$\phi^2\f$, used by the
 * CCZ4 formulation of Einstein's equations.
 *
 * \details If \f$ \gamma_{ij}\f$ is the spatial metric, then \f$\phi^2\f$ is
 * computed as
 *
 * \f{align}
 *     \phi^2 &= (det(\gamma_{ij}))^{-1/6}
 * \f}
 */
template <typename DataType>
void phi_squared(const gsl::not_null<Scalar<DataType>*> phi_squared,
                 const Scalar<DataType>& phi) noexcept;

template <typename DataType>
Scalar<DataType> phi_squared(const Scalar<DataType>& phi) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item for the square of the conformal factor, \f$\phi^2\f$,
 * used by the CCZ4 formulation of Einstein's equations.
 *
 * \details See `phi_squared()`. Can be retrieved using
 * `CCZ4::Tags::PhiSquared`.
 */
template <typename DataType>
struct PhiSquaredCompute : PhiSquared<DataType>, db::ComputeTag {
  using argument_tags = tmpl::list<Phi<DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const Scalar<DataType>&) noexcept>(
      &phi_squared<DataType>);

  using base = PhiSquared<DataType>;
};
}  // namespace Tags
}  // namespace CCZ4
