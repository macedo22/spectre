// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/CCZ4/ATilde.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace CCZ4 {
template <size_t Dim, typename Frame, typename DataType>
void a_tilde(const gsl::not_null<tnsr::ij<DataType, Dim, Frame>*> a_tilde,
             const Scalar<DataType>& phi_squared,
             const tnsr::ii<DataType, Dim, Frame>& extrinsic_curvature,
             const Scalar<DataType>& trace_extrinsic_curvature,
             const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept {
  ::TensorExpressions::evaluate<ti_i, ti_j>(
      a_tilde, phi_squared() * (extrinsic_curvature(ti_i, ti_j) -
                                (1.0 / 3) * trace_extrinsic_curvature() *
                                    spatial_metric(ti_i, ti_j)));
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::ij<DataType, Dim, Frame> a_tilde(
    const Scalar<DataType>& phi_squared,
    const tnsr::ii<DataType, Dim, Frame>& extrinsic_curvature,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept {
  tnsr::ij<DataType, Dim, Frame> a_tilde{};
  ::CCZ4::a_tilde(make_not_null(&a_tilde), phi_squared, extrinsic_curvature,
                  trace_extrinsic_curvature, spatial_metric);
  return a_tilde;
}
}  // namespace CCZ4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void CCZ4::a_tilde(                                            \
      const gsl::not_null<tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>*> \
          a_tilde,                                                        \
      const Scalar<DTYPE(data)>& phi_squared,                             \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          extrinsic_curvature,                                            \
      const Scalar<DTYPE(data)>& trace_extrinsic_curvature,               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          spatial_metric) noexcept;                                       \
  template tnsr::ij<DTYPE(data), DIM(data), FRAME(data)> CCZ4::a_tilde(   \
      const Scalar<DTYPE(data)>& phi_squared,                             \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          extrinsic_curvature,                                            \
      const Scalar<DTYPE(data)>& trace_extrinsic_curvature,               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Inertial, Frame::Grid))

#undef FRAME
#undef DTYPE
#undef DIM
#undef INSTANTIATE
