// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "Evolution/Systems/Ccz4/DerivLapse.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim, typename Frame, typename DataType>
void grad_grad_lapse(
    const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*> grad_grad_lapse_,
    const Scalar<DataType>& lapse,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& field_a,
    const tnsr::ij<DataType, Dim, Frame>& d_field_a) noexcept {
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      grad_grad_lapse_->get(i, j) =
          field_a.get(i) * field_a.get(j) +
          0.5 * (d_field_a.get(i, j) + d_field_a.get(j, i));
      for (size_t k = 0; k < Dim; ++k) {
        grad_grad_lapse_->get(i, j) -=
            christoffel_second_kind.get(k, i, j) * field_a.get(k);
      }
      grad_grad_lapse_->get(i, j) *= lapse.get();
    }
  }
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::ii<DataType, Dim, Frame> grad_grad_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& field_a,
    const tnsr::ij<DataType, Dim, Frame>& d_field_a) noexcept {
  tnsr::ii<DataType, Dim, Frame> grad_grad_lapse_{};
  grad_grad_lapse(make_not_null(&grad_grad_lapse_), lapse,
                  christoffel_second_kind, field_a, d_field_a);
  return grad_grad_lapse_;
}

template <size_t Dim, typename Frame, typename DataType>
void divergence_lapse(
    const gsl::not_null<Scalar<DataType>*> div_lapse,
    const Scalar<DataType>& conformal_factor,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_metric,
    const tnsr::ii<DataType, Dim, Frame>& grad_grad_lapse) noexcept {
  destructive_resize_components(div_lapse, get_size(get(conformal_factor)));
  for (auto& component : *div_lapse) {
    component = 0.0;
  }

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      if (i == j) {
        div_lapse->get() +=
            inverse_conformal_metric.get(i, j) * grad_grad_lapse.get(i, j);
      } else {
        div_lapse->get() += 2.0 * inverse_conformal_metric.get(i, j) *
                            grad_grad_lapse.get(i, j);
      }
    }
  }
  div_lapse->get() *= square(conformal_factor.get());
}

template <size_t Dim, typename Frame, typename DataType>
Scalar<DataType> divergence_lapse(
    const Scalar<DataType>& conformal_factor,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_metric,
    const tnsr::ii<DataType, Dim, Frame>& grad_grad_lapse) noexcept {
  Scalar<DataType> div_lapse{};
  divergence_lapse(make_not_null(&div_lapse), conformal_factor,
                   inverse_conformal_metric, grad_grad_lapse);
  return div_lapse;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void Ccz4::grad_grad_lapse(                                    \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*> \
          grad_grad_lapse_,                                               \
      const Scalar<DTYPE(data)>& lapse,                                   \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          christoffel_second_kind,                                        \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_a,        \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_field_a) noexcept;                                            \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                  \
  Ccz4::grad_grad_lapse(                                                  \
      const Scalar<DTYPE(data)>& lapse,                                   \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          christoffel_second_kind,                                        \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_a,        \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_field_a) noexcept;                                            \
  template void Ccz4::divergence_lapse(                                   \
      const gsl::not_null<Scalar<DTYPE(data)>*> div_lapse,                \
      const Scalar<DTYPE(data)>& conformal_factor,                        \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_conformal_metric,                                       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          grad_grad_lapse) noexcept;                                      \
  template Scalar<DTYPE(data)> Ccz4::divergence_lapse(                    \
      const Scalar<DTYPE(data)>& conformal_factor,                        \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_conformal_metric,                                       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          grad_grad_lapse) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
