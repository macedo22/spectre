// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4 {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the gradient of the gradient of the lapse.
 *
 * \details Computes the gradient of the gradient as:
 * \f{align}
 *     \nabla_i \nabla_j \alpha &= \alpha A_i A_j -
 *                 \alpha \Gamma^k{}_{ij} A_k + \alpha \partial_{(i} A_{j)}
 * \f}
 * where \f$\alpha\f$, \f$\Gamma^k{}_{ij}\f$, \f$A_i\f$, and
 * \f$\partial_j A_i\f$ are the lapse, spatial christoffel symbols of the second
 * kind, the CCZ4 auxiliary variable defined by `Ccz4::Tags::FieldA`, and its
 * spatial derivative, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
void grad_grad_lapse(
    const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*> grad_grad_lapse_,
    const Scalar<DataType>& lapse,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& field_a,
    const tnsr::ij<DataType, Dim, Frame>& d_field_a) noexcept;

template <size_t Dim, typename Frame, typename DataType>
tnsr::ii<DataType, Dim, Frame> grad_grad_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& field_a,
    const tnsr::ij<DataType, Dim, Frame>& d_field_a) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item to get the gradient of the gradient of the lapse.
 *
 * \details See `grad_grad_lapse()`. Can be retrieved using
 * `Ccz4::Tags::GradGradLapse`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct GradGradLapseCompute : Ccz4::Tags::GradGradLapse<Dim, Frame, DataType>,
                              db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataType>,
      gr::Tags::SpatialChristoffelSecondKind<Dim, Frame, DataType>,
      FieldA<Dim, Frame, DataType>,
      ::Tags::deriv<FieldA<Dim, Frame, DataType>, tmpl::size_t<Dim>, Frame>>;

  using return_type = tnsr::ii<DataType, Dim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*>,
      const Scalar<DataType>&, const tnsr::Ijj<DataType, Dim, Frame>&,
      const tnsr::i<DataType, Dim, Frame>&,
      const tnsr::ij<DataType, Dim, Frame>&) noexcept>(
      &Ccz4::grad_grad_lapse<Dim, Frame, DataType>);

  using base = Ccz4::Tags::GradGradLapse<Dim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace Ccz4
