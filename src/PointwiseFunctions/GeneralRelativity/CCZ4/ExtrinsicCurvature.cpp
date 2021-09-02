// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/CCZ4/ExtrinsicCurvature.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace CCZ4 {
template <size_t SpatialDim, typename Frame, typename DataType>
void extrinsic_curvature(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        extrinsic_curvature) noexcept {
  // TODO
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = 0; j < SpatialDim; j++) {
      (*extrinsic_curvature).get(i, j) = 1.0;
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature() noexcept {
  tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature{};
  ::CCZ4::extrinsic_curvature(make_not_null(&extrinsic_curvature));
  return extrinsic_curvature;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void trace_extrinsic_curvature(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  ::TensorExpressions::evaluate(
      trace_extrinsic_curvature,
      extrinsic_curvature(ti_i, ti_j) * inverse_spatial_metric(ti_I, ti_J));
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> trace_extrinsic_curvature(
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  Scalar<DataType> trace_extrinsic_curvature{};
  ::CCZ4::trace_extrinsic_curvature(make_not_null(&trace_extrinsic_curvature),
                                    extrinsic_curvature,
                                    inverse_spatial_metric);
  return trace_extrinsic_curvature;
}
}  // namespace CCZ4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void CCZ4::extrinsic_curvature(                                 \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*>  \
          extrinsic_curvature) noexcept;                                   \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                   \
  CCZ4::extrinsic_curvature() noexcept;                                    \
  template void CCZ4::trace_extrinsic_curvature(                           \
      const gsl::not_null<Scalar<DTYPE(data)>*> trace_extrinsic_curvature, \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                 \
          extrinsic_curvature,                                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric) noexcept;                                \
  template Scalar<DTYPE(data)> CCZ4::trace_extrinsic_curvature(            \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                 \
          extrinsic_curvature,                                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
