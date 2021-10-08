// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/Z4Constraint.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim, typename Frame, typename DataType>
void spatial_z4_constraint(
    const gsl::not_null<tnsr::i<DataType, Dim, Frame>*> result,
    const gsl::not_null<tnsr::I<DataType, Dim, Frame>*> buffer,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::I<DataType, Dim, Frame>&
        contracted_conformal_christoffel_second_kind,
    const tnsr::I<DataType, Dim, Frame>& gamma_hat) {
  destructive_resize_components(result,
                                get_size(get<0, 0>(conformal_spatial_metric)));
  destructive_resize_components(buffer,
                                get_size(get<0, 0>(conformal_spatial_metric)));

  ::TensorExpressions::evaluate<ti_J>(
      buffer,
      gamma_hat(ti_J) - contracted_conformal_christoffel_second_kind(ti_J));

  ::TensorExpressions::evaluate<ti_i>(
      result, 0.5 * (conformal_spatial_metric(ti_i, ti_j) * (*buffer)(ti_J)));
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::i<DataType, Dim, Frame> spatial_z4_constraint(
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::I<DataType, Dim, Frame>&
        contracted_conformal_christoffel_second_kind,
    const tnsr::I<DataType, Dim, Frame>& gamma_hat) {
  tnsr::i<DataType, Dim, Frame> result{};
  tnsr::I<DataType, Dim, Frame> buffer{};
  spatial_z4_constraint(
      make_not_null(&result), make_not_null(&buffer), conformal_spatial_metric,
      contracted_conformal_christoffel_second_kind, gamma_hat);
  return result;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                             \
  template void Ccz4::spatial_z4_constraint(                             \
      const gsl::not_null<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                        \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*> \
          buffer,                                                        \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&               \
          conformal_spatial_metric,                                      \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>&                \
          contracted_conformal_christoffel_second_kind,                  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& gamma_hat);    \
  template tnsr::i<DTYPE(data), DIM(data), FRAME(data)>                  \
  Ccz4::spatial_z4_constraint(                                           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&               \
          conformal_spatial_metric,                                      \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>&                \
          contracted_conformal_christoffel_second_kind,                  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& gamma_hat);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
